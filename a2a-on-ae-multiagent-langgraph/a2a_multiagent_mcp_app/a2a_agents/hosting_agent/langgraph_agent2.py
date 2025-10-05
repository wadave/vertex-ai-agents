# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module defines the HostingAgent, a central routing agent.

The HostingAgent is responsible for managing connections to various specialized
agents (like Weather and Cocktail agents), delegating user requests to the
appropriate agent, and returning the results. It uses LangGraph to create a
ReAct-style agent that can reason about which tool (i.e., which specialized
agent) to call based on the user's query.

The agent's logic is primarily defined in the `root_instruction` method, which
dynamically generates a system prompt to guide the underlying language model.
This allows the agent to handle greetings, capability questions, and task
delegation appropriately.
"""
import asyncio
import logging
import os
import uuid
from typing import Annotated

import httpx
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import InjectedState, create_react_agent

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import (
    AgentCard,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TextPart,
    TransportProtocol,
)
from remote_connection import RemoteAgentConnections, TaskUpdateCallback

load_dotenv()

logger = logging.getLogger(__name__)
memory = MemorySaver()

# Set to True to enable detailed debug logging
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"


class HostingAgent:
    """The host agent.

    This is the agent responsible for choosing which remote agents to send
    tasks to and coordinate their work.

    Attributes:
        task_callback: An optional callback function to be invoked when a task's
          status is updated.
        httpx_client: An asynchronous HTTP client for making requests to remote
          agents.
        client_factory: A factory for creating clients to communicate with remote
          agents based on their transport protocols.
        remote_agent_connections: A dictionary mapping agent names to their
          `RemoteAgentConnections` instances, which manage communication.
        cards: A dictionary mapping agent names to their `AgentCard` objects,
          which contain metadata about each agent's capabilities.
        agents: A formatted string listing the available agents and their skills,
          used for populating the agent's prompt.
        remote_agent_addresses: A list of base URLs for the remote agents that
          this hosting agent can connect to.
        _cards_loaded: A boolean flag indicating whether the agent cards for all
          remote agents have been successfully loaded.
    """

    def __init__(
        self,
        remote_agent_addresses: list[str],
        http_client: httpx.AsyncClient,
        task_callback: TaskUpdateCallback | None = None,
    ):
        self.task_callback = task_callback
        self.httpx_client = http_client
        config = ClientConfig(
            httpx_client=self.httpx_client,
            supported_transports=[
                TransportProtocol.jsonrpc,
                TransportProtocol.http_json,
            ],
        )
        client_factory = ClientFactory(config)
        self.client_factory = client_factory
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self.agents: str = ""
        self.remote_agent_addresses = remote_agent_addresses
        self._cards_loaded = False

    async def init_remote_agent_addresses(self, remote_agent_addresses: list[str]):
        """Initialize connections to remote agents by fetching their cards."""
        logger.info(
            f"Fetching agent cards from {len(remote_agent_addresses)} addresses..."
        )
        async with asyncio.TaskGroup() as task_group:
            for address in remote_agent_addresses:
                task_group.create_task(self.retrieve_card(address))

        self._cards_loaded = True
        logger.info(
            f"All agent cards loaded! Agents available: {list(self.cards.keys())}"
        )

    async def retrieve_card(self, address: str):
        """Retrieve an agent card from a remote agent address."""
        card_resolver = A2ACardResolver(
            self.httpx_client, base_url=address, agent_card_path="/v1/card"
        )
        card = await card_resolver.get_agent_card()
        logger.info(f"Retrieved card for {card.name} from {address}")
        self.register_agent_card(card)

    def register_agent_card(self, card: AgentCard):
        """Register an agent card and create a connection to it."""
        remote_connection = RemoteAgentConnections(self.client_factory, card)
        self.remote_agent_connections[card.name] = remote_connection
        self.cards[card.name] = card
        self._update_agents_list()

    def _update_agents_list(self):
        """Update the formatted agents list used in the prompt."""
        agent_info = []
        for card in self.cards.values():
            info = f"- {card.name}: {card.description}"
            if card.skills:
                skills = ", ".join([s.description for s in card.skills])
                info += f"\n  Skills: {skills}"
            agent_info.append(info)
        self.agents = "\n".join(agent_info) if agent_info else "No agents available yet"

    def create_agent(self):
        """Create a LangGraph ReAct agent.

        Returns:
            CompiledStateGraph that can be used for streaming execution.
        """
        GOOGLE_GENAI_MODEL = os.getenv("GOOGLE_GENAI_MODEL", "gemini-2.5-flash")
        model = ChatVertexAI(
            model=GOOGLE_GENAI_MODEL,
            temperature=0,
        )

        return create_react_agent(
            model,
            tools=[self.send_message],
            checkpointer=memory,
            prompt=self.root_instruction,
        )

    def root_instruction(self, state: dict) -> str:
        """Generate the system prompt for the agent.

        The prompt changes dynamically based on the conversation state:
        - If a tool response was just received, instructs to relay it
        - Otherwise, instructs to route the request to the appropriate agent

        Args:
            state: The graph state containing messages and other context

        Returns:
            The system prompt string
        """
        agents_list = self.agents if self.agents else "Loading agents..."

        # Check if we just received a tool response
        has_tool_response = False
        tool_response_content = ""
        if state.get("messages"):
            messages = state["messages"]
            if messages:
                last_msg = messages[-1]
                msg_type = type(last_msg).__name__
                if hasattr(last_msg, "content"):
                    content = str(last_msg.content)
                    if msg_type == "ToolMessage":
                        has_tool_response = True
                        tool_response_content = content

        if DEBUG_MODE:
            logger.debug(
                f"Prompt function called - Agents: {len(self.cards)}, "
                f"Messages: {len(state.get('messages', []))}, "
                f"Has tool response: {has_tool_response}"
            )

        # If we just got a tool response, tell the agent to STOP and return it
        if has_tool_response:
            return f"""You just received this response from a specialized agent:

"{tool_response_content}"

YOUR ONLY TASK: Return the above response EXACTLY as-is to the user. Do not modify it, do not add commentary, just relay it."""

        # Extract the original user question to include in the prompt
        user_question = ""
        if state.get("messages"):
            for msg in state["messages"]:
                if type(msg).__name__ == "HumanMessage":
                    user_question = str(msg.content) if hasattr(msg, "content") else ""
                    break

        # Check if user is greeting, being conversational, or asking about capabilities
        should_respond_directly = False
        if user_question:
            lower_question = user_question.lower().strip()

            # Greetings and conversational phrases
            greetings = ["hi", "hello", "hey", "greetings", "howdy"]
            conversational = [
                "how are you",
                "how is it going",
                "how's it going",
                "what's up",
                "how do you do",
                "nice to meet you",
            ]
            is_greeting = lower_question in greetings or any(
                lower_question.startswith(g + " ") or lower_question.startswith(g + ",")
                for g in greetings
            )
            is_conversational = any(
                phrase in lower_question for phrase in conversational
            )

            # Capability questions
            capability_keywords = [
                "what can you",
                "what do you",
                "what are you",
                "your capabilities",
                "help me",
                "what can i",
                "what services",
                "what features",
            ]
            is_capability_question = any(
                keyword in lower_question for keyword in capability_keywords
            )

            should_respond_directly = (
                is_greeting or is_conversational or is_capability_question
            )

        if should_respond_directly:
            return f"""The user is greeting you, being conversational, or asking about your capabilities.

AVAILABLE AGENTS AND THEIR CAPABILITIES:
{agents_list}

YOUR TASK: Respond naturally and helpfully. If it's a greeting or conversational question (like "how are you?"), respond warmly and then introduce your capabilities.

Explain that you can help with ALL of the following:
1. Weather-related questions: forecasts, alerts, temperature, etc. (via Weather Agent)
2. Cocktail/drink-related questions: recipes, ingredients, drink information, etc. (via Cocktail Agent)

Be friendly, conversational, and comprehensive. Mention BOTH agent capabilities.
DO NOT call any tools - respond directly."""

        return f"""You are a routing agent that delegates requests to specialized agents.

AVAILABLE AGENTS:
{agents_list}

THE USER ASKED: "{user_question}"

YOUR WORKFLOW (EXACTLY 2 STEPS):
Step 1: Call send_message ONCE with:
   - agent_name: Choose based on topic:
     * Weather/temperature/forecast → "Weather Agent - langgraph"
     * Cocktail/drink/recipe → "Cocktail Agent - LangGraph"
   - message: COPY THE EXACT USER QUESTION: "{user_question}"
     DO NOT rephrase or change it!

Step 2: After receiving the tool's response, return it to the user
   - Just relay what you received
   - STOP - do not call tools again

CRITICAL: The message parameter MUST be EXACTLY: "{user_question}"
Do NOT rephrase, modify, or create a different question!
"""

    def check_state(self, state: dict):
        """Check the current state to determine the active agent.

        In LangGraph, state is passed directly as a dict.
        """
        if (
            "context_id" in state
            and "session_active" in state
            and state["session_active"]
            and "agent" in state
        ):
            return {"active_agent": f"{state['agent']}"}
        return {"active_agent": "None"}

    def list_remote_agents(self) -> str:
        """List the available remote agents you can use to delegate the task.

        Returns:
            A formatted string describing each available agent and their capabilities.
        """
        if not self.remote_agent_connections:
            return "No remote agents are currently available."

        remote_agent_info = []
        for card in self.cards.values():
            agent_desc = f"- **{card.name}**: {card.description}"
            if card.skills:
                skills_desc = ", ".join([s.description for s in card.skills])
                agent_desc += f"\n  Skills: {skills_desc}"
            remote_agent_info.append(agent_desc)

        return "Available agents:\n" + "\n".join(remote_agent_info)

    async def send_message(
        self, agent_name: str, message: str, state: Annotated[dict, InjectedState]
    ) -> str:
        """Send a message to a remote agent and get their response.

        ROUTING RULES:
        - For weather/temperature/forecast questions → "Weather Agent - langgraph"
        - For cocktail/drink/recipe questions → "Cocktail Agent - LangGraph"

        Args:
            agent_name: The exact name of the remote agent
            message: The user's original question (should not be rephrased)
            state: The graph state (automatically injected by LangGraph)

        Returns:
            The response from the remote agent

        Raises:
            ValueError: If the agent_name is not found or client is unavailable
        """
        if DEBUG_MODE:
            logger.debug(
                f"send_message called - Agent: {agent_name}, Message: {message[:50]}..."
            )

        if agent_name not in self.remote_agent_connections:
            available = list(self.remote_agent_connections.keys())
            raise ValueError(f'Agent "{agent_name}" not found. Available: {available}')

        state["agent"] = agent_name
        client = self.remote_agent_connections[agent_name]
        if not client:
            raise ValueError(f"Client not available for {agent_name}")
        task_id = state.get("task_id", None)
        context_id = state.get("context_id", None)
        message_id = state.get("message_id", None)
        task: Task
        if not message_id:
            message_id = str(uuid.uuid4())

        request_message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=message))],
            message_id=message_id,
            context_id=context_id,
            task_id=task_id,
        )
        response = await client.send_message(request_message)
        if isinstance(response, Message):
            parts_text = convert_parts_simple(response.parts)
            result = "\n".join(parts_text) if parts_text else "No response"

            if DEBUG_MODE:
                logger.debug(f"Message response from {agent_name}: {result[:100]}...")

            return result

        task: Task = response
        # Assume completion unless a state returns that isn't complete
        state["session_active"] = task.status.state not in [
            TaskState.completed,
            TaskState.canceled,
            TaskState.failed,
            TaskState.unknown,
        ]
        if task.context_id:
            state["context_id"] = task.context_id
        state["task_id"] = task.id

        if task.status.state == TaskState.input_required:
            # Mark as needing user input
            state["needs_user_input"] = True
        elif task.status.state == TaskState.canceled:
            # Open question, should we return some info for cancellation instead
            raise ValueError(f"Agent {agent_name} task {task.id} is cancelled")
        elif task.status.state == TaskState.failed:
            # Raise error for failure
            raise ValueError(f"Agent {agent_name} task {task.id} failed")

        response_parts = []
        if task.status.message:
            # Assume the information is in the task message.
            response_parts.extend(convert_parts_simple(task.status.message.parts))
        if task.artifacts:
            for artifact in task.artifacts:
                response_parts.extend(convert_parts_simple(artifact.parts))

        result = (
            "\n".join(response_parts)
            if response_parts
            else f"Task {task.id} completed with status {task.status.state}"
        )

        if DEBUG_MODE:
            logger.debug(
                f"Task response from {agent_name} (State: {task.status.state}): {result[:100]}..."
            )

        return result


def convert_parts_simple(parts: list[Part]) -> list[str]:
    """Convert A2A Parts to simple string responses for LangGraph.

    This is a simplified version that works without ToolContext.
    For file handling, we just include basic info rather than saving artifacts.
    """
    result = []
    for part in parts:
        if part.root.kind == "text":
            result.append(part.root.text)
        elif part.root.kind == "data":
            # Convert data dict to string representation
            result.append(str(part.root.data))
        elif part.root.kind == "file":
            # For files, just note that a file was received
            file_name = part.root.file.name
            mime_type = part.root.file.mime_type
            result.append(f"[File received: {file_name} ({mime_type})]")
        else:
            result.append(f"Unknown type: {part.kind}")
    return result


async def get_root_agent(httpx_client: httpx.AsyncClient | None = None):
    """Create and initialize the root hosting agent.

    This function creates a HostingAgent, fetches all remote agent cards,
    and returns a configured LangGraph agent ready to process requests.

    Args:
        httpx_client: Optional HTTP client for making requests to remote agents

    Returns:
        A compiled LangGraph agent ready for streaming execution
    """
    hosting_agent = HostingAgent(
        remote_agent_addresses=[
            os.getenv("CT_AGENT_URL", "http://localhost:10002"),
            os.getenv("WEA_AGENT_URL", "http://localhost:10001"),
        ],
        http_client=httpx_client,
    )

    # Wait for agent cards to be loaded before processing any queries
    await hosting_agent.init_remote_agent_addresses(
        hosting_agent.remote_agent_addresses
    )

    # Create the LangGraph agent with fully loaded cards
    return hosting_agent.create_agent()
