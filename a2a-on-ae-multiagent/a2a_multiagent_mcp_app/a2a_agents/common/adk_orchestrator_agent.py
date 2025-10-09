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
# Author: Dave Wang
import asyncio
import base64
import json
import logging
import uuid

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import (
    AgentCard,
    DataPart,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TextPart,
    TransportProtocol,
)
from dotenv import load_dotenv
from google import adk
from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from common.remote_connection import RemoteAgentConnections, TaskUpdateCallback

# from google.adk.models.lite_llm import LiteLlm

logger = logging.getLogger(__name__)

load_dotenv()


async def auto_save_session_to_memory_callback(callback_context: CallbackContext):
    """
    Callback to save conversation session to Vertex AI Memory Bank.

    This callback is triggered after the agent completes processing.
    It extracts conversation events from the session and sends them to
    the Memory Bank service for processing. The service generates semantic
    memories that can be retrieved in future conversations.

    Args:
        callback_context: The callback context containing session and memory service.
    """
    session = callback_context._invocation_context.session
    memory_service = callback_context._invocation_context.memory_service

    logging.info(
        f"Saving session {session.id} to memory bank for user_id={session.user_id}"
    )

    try:
        await memory_service.add_session_to_memory(session)
        logging.info(f"Memory generation completed for session {session.id}")
    except Exception as e:
        logging.error(
            f"Memory generation failed for session {session.id}: {e}",
            exc_info=True,
        )


class AdkOrchestratorAgent:
    """The orchestrator agent.

    This is the agent responsible for choosing which remote agents to send
    tasks to and coordinate their work.
    """

    def __init__(
        self,
        remote_agent_addresses: list[str],
        http_client: httpx.AsyncClient,
        task_callback: TaskUpdateCallback | None = None,
    ):
        """Initializes the OrchestratorAgent.

        Args:
            remote_agent_addresses: A list of remote agent addresses.
            http_client: An httpx client.
            task_callback: A callback for task updates.
        """
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
        self._init_task = None
        self._remote_agent_addresses = remote_agent_addresses

    async def init_remote_agent_addresses(self, remote_agent_addresses: list[str]):
        """Initializes the remote agent addresses.

        Args:
            remote_agent_addresses: A list of remote agent addresses.
        """
        # Use asyncio.gather for Python 3.10 compatibility (TaskGroup is 3.11+)
        tasks = [self.retrieve_card(address) for address in remote_agent_addresses]
        await asyncio.gather(*tasks)
        # Once completed the self.agents string is set and the remote
        # connections are established.

    async def retrieve_card(self, address: str):
        """Retrieves the agent card from the given address.

        Args:
            address: The address of the agent.
        """
        card_resolver = A2ACardResolver(
            self.httpx_client, base_url=address, agent_card_path="/v1/card"
        )
        card = await card_resolver.get_agent_card()

        logger.info(f"Retrieved card for {card.name} from {address}")
        self.register_agent_card(card)

    def register_agent_card(self, card: AgentCard):
        """Registers the agent card.

        Args:
            card: The agent card to register.
        """
        remote_connection = RemoteAgentConnections(self.client_factory, card)
        self.remote_agent_connections[card.name] = remote_connection
        self.cards[card.name] = card
        agent_info = []
        for ra in self.list_remote_agents():
            agent_info.append(json.dumps(ra))
        self.agents = "\n".join(agent_info)

    def create_agent(self) -> Agent:
        """Creates the orchestrator agent."""
        return Agent(
            model="gemini-2.5-flash",
            name="orchestrator_agent",
            instruction=self.root_instruction,
            before_model_callback=self.before_model_callback,
            description=(
                "This agent orchestrates the decomposition of the user request into"
                " tasks that can be performed by the child agents."
            ),
            tools=[
                self.list_remote_agents,
                self.send_message,
                adk.tools.preload_memory_tool.PreloadMemoryTool(),
            ],
            after_agent_callback=auto_save_session_to_memory_callback,
        )

    def root_instruction(self, context: ReadonlyContext) -> str:
        current_agent = self.check_state(context)
        return f"""You are an expert delegator that can delegate the user request to the
appropriate remote agents.

Discovery:
- You can use `list_remote_agents` to list the available remote agents you
can use to delegate the task.

Execution:
- For actionable requests, you can use `send_message` to interact with remote agents to take action.
- You could use tools to check previous conversations, and relay information in response to user queries.

Be sure to include the remote agent name when you respond to the user.

Please rely on tools to address the request, and don't make up the response. If you are not sure, please ask the user for more details.
Focus on the most recent parts of the conversation primarily.

Agents:
{self.agents}

Current agent: {current_agent["active_agent"]} """

    def check_state(self, context: ReadonlyContext) -> dict[str, str]:
        """Checks the state of the agent.

        Args:
            context: The readonly context.

        Returns:
            A dictionary with the active agent.
        """
        state = context.state
        if (
            "context_id" in state
            and "session_active" in state
            and state["session_active"]
            and "agent" in state
        ):
            return {"active_agent": f"{state['agent']}"}
        return {"active_agent": "None"}

    def before_model_callback(self, callback_context: CallbackContext, llm_request):
        """A callback that is called before the model is called.

        Args:
            callback_context: The callback context.
            llm_request: The llm request.
        """
        state = callback_context.state
        if "session_active" not in state or not state["session_active"]:
            state["session_active"] = True

    def list_remote_agents(self) -> list[dict[str, str]]:
        """List the available remote agents you can use to delegate the task."""
        if not self.remote_agent_connections:
            return []

        remote_agent_info = []
        for card in self.cards.values():
            remote_agent_info.append(
                {"name": card.name, "description": card.description}
            )
        return remote_agent_info

    async def send_message(
        self,
        agent_name: str,
        message: str,
        tool_context: ToolContext,
    ):
        """Sends a task either streaming (if supported) or non-streaming.

        This will send a message to the remote agent named agent_name.

        Args:
          agent_name: The name of the agent to send the task to.
          message: The message to send to the agent for the task.
          tool_context: The tool context this method runs in.

        Yields:
          A dictionary of JSON data.
        """
        if agent_name not in self.remote_agent_connections:
            raise ValueError(f"Agent {agent_name} not found")
        state = tool_context.state
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
            logger.info("Got message object from remote agent")
            logger.debug(f"Message content: {response}")
            return await convert_parts(response.parts, tool_context)
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
            # Force user input back
            tool_context.actions.skip_summarization = True
            tool_context.actions.escalate = True
        elif task.status.state == TaskState.canceled:
            # Open question, should we return some info for cancellation instead
            raise ValueError(f"Agent {agent_name} task {task.id} is cancelled")
        elif task.status.state == TaskState.failed:
            # Raise error for failure
            raise ValueError(f"Agent {agent_name} task {task.id} failed")
        response = []
        if task.status.message:
            # Assume the information is in the task message.
            response.extend(
                await convert_parts(task.status.message.parts, tool_context)
            )
        if task.artifacts:
            for artifact in task.artifacts:
                response.extend(await convert_parts(artifact.parts, tool_context))
        return response


async def convert_parts(parts: list[Part], tool_context: ToolContext) -> list:
    """Converts a list of parts.

    Args:
        parts: The list of parts to convert.
        tool_context: The tool context.

    Returns:
        A list of converted parts.
    """
    rval = []
    for p in parts:
        rval.append(await convert_part(p, tool_context))
    return rval


async def convert_part(part: Part, tool_context: ToolContext) -> str | DataPart | dict:
    """Converts a part.

    Args:
        part: The part to convert.
        tool_context: The tool context.

    Returns:
        The converted part (string, DataPart, or dict).
    """
    if part.root.kind == "text":
        return part.root.text
    if part.root.kind == "data":
        return part.root.data
    if part.root.kind == "file":
        # Repackage A2A FilePart to google.genai Blob
        # Currently not considering plain text as files
        file_id = part.root.file.name
        file_bytes = base64.b64decode(part.root.file.bytes)
        file_part = types.Part(
            inline_data=types.Blob(mime_type=part.root.file.mime_type, data=file_bytes)
        )
        await tool_context.save_artifact(file_id, file_part)
        tool_context.actions.skip_summarization = True
        tool_context.actions.escalate = True
        return DataPart(data={"artifact-file-id": file_id})
    return f"Unknown type: {part.kind}"


async def get_orchestrator_agent(
    remote_agent_addresses: list[str], httpx_client: httpx.AsyncClient | None = None
) -> Agent:
    """Gets the orchestrator agent.

    Args:
        remote_agent_addresses: A list of remote agent addresses.
        httpx_client: An httpx client.

    Returns:
        The orchestrator agent.
    """
    orchestrator_agent_wrapper = AdkOrchestratorAgent(
        remote_agent_addresses=remote_agent_addresses,
        http_client=httpx_client,
    )

    # Initialize remote agents before creating the agent
    # This ensures agents are available when the orchestrator is first used
    await orchestrator_agent_wrapper.init_remote_agent_addresses(
        remote_agent_addresses
    )

    return orchestrator_agent_wrapper.create_agent()
