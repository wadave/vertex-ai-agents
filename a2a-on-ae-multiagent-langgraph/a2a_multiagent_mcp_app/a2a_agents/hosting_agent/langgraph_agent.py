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

import asyncio
import logging
import os
import uuid
from typing import Annotated, Literal

import httpx
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import InjectedState, create_react_agent, ToolNode
from pydantic import BaseModel, Field

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
        """Create a LangGraph agent with explicit control flow.

        Returns:
            CompiledStateGraph that can be used for streaming execution.
        """
        GOOGLE_GENAI_MODEL = os.getenv("GOOGLE_GENAI_MODEL", "gemini-2.5-flash")
        model = ChatVertexAI(
            model=GOOGLE_GENAI_MODEL,
            temperature=0,
        )

        # Bind tools to the model
        model_with_tools = model.bind_tools([self.send_message])

        # Create agent node
        def agent_node(state: MessagesState):
            """Agent decides whether to use tools or respond directly."""
            from langchain_core.messages import SystemMessage

            # Get the system prompt
            system_msg = SystemMessage(content=self.root_instruction(state))
            messages = [system_msg] + state["messages"]

            response = model_with_tools.invoke(messages)
            return {"messages": [response]}

        # Create tool node
        tool_node = ToolNode([self.send_message])

        # Define routing logic
        def should_continue(state: MessagesState):
            """Decide whether to continue to tools or end."""
            messages = state["messages"]
            last_message = messages[-1]

            # If there are tool calls, go to tools
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            # Otherwise, end
            return END

        # Build the graph
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue, ["tools", END])
        workflow.add_edge("tools", "agent")  # After tools, go back to agent ONCE

        return workflow.compile(checkpointer=memory)

    def root_instruction(self, state: dict) -> str:
        """Generate the system prompt for the agent.

        Args:
            state: The graph state containing messages and other context

        Returns:
            The system prompt string
        """
        agents_list = self.agents if self.agents else "Loading agents..."

        # Get current active agent if any
        current_agent = self.check_state(state).get("active_agent", "None")

        if DEBUG_MODE:
            logger.debug(
                f"Prompt function - Agents: {len(self.cards)}, "
                f"Messages: {len(state.get('messages', []))}"
            )

        return f"""You are a helpful assistant that can answer questions about weather and cocktails by delegating to specialized agents.

CRITICAL RULES:
1. For greetings or small talk (hi, hello, how are you, what's up), respond directly WITHOUT using any tools.
2. For questions about your capabilities (what can you do, help), describe both weather and cocktail capabilities WITHOUT using tools.
3. ONLY use tools for actual weather or cocktail questions.
4. When you use a tool, use it ONLY ONCE and then return the response.
5. NEVER call the same tool multiple times in a row.

Available Agents:
{agents_list}

Current active agent: {current_agent}

Example interactions:
User: "hi" → You: "Hello! I can help with weather forecasts and cocktail recipes. What would you like to know?"
User: "how are you" → You: "I'm doing well, thank you! I can assist with weather information and cocktail recipes."
User: "weather in LA" → You: Use send_message tool with Weather Agent ONCE, then return the result
User: "margarita recipe" → You: Use send_message tool with Cocktail Agent ONCE, then return the result
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
