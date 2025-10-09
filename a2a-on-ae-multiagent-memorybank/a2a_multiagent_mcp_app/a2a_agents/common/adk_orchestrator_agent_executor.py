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
import logging
import os
from abc import ABC, abstractmethod
from typing import NoReturn

import httpx

# A2A
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Role, TaskState, TextPart, UnsupportedOperationError
from a2a.utils import new_agent_text_message
from a2a.utils.errors import ServerError
from dotenv import load_dotenv
from google.adk import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService, VertexAiSessionService
from google.genai import types

from common.adk_orchestrator_agent import get_orchestrator_agent
from common.auth_utils import GoogleAuth
from common.adk_base_mcp_agent_executor import PersistentVertexAiMemoryBankService

# Set logging
logging.getLogger().setLevel(logging.INFO)
load_dotenv()


class AdkOrchestratorAgentExecutor(AgentExecutor, ABC):
    """Base abstract class for orchestrator agent executors that bridge A2A protocol
    with ADK agents.

    The executor handles:
    1. Protocol translation (A2A messages to/from agent format)
    2. Task lifecycle management (submitted -> working -> completed)
    3. Session management for multi-turn conversations
    4. Error handling and recovery
    5. Agent engine configuration with custom memory topics
    """

    def __init__(
        self, remote_agent_addresses: list[str], agent_engine_id: str = None
    ) -> None:
        """Initialize with lazy loading pattern.

        Args:
            remote_agent_addresses: A list of remote agent addresses.
            agent_engine_id: Optional agent engine ID for memory bank. If not
              provided, creates a new one.
        """
        self.remote_agent_addresses = remote_agent_addresses
        self.agent = None
        self.runner = None
        self.agent_engine_id = agent_engine_id

        if self.agent_engine_id is None:
            self.agent_engine_id = self.get_agent_engine()

    @abstractmethod
    def get_agent_engine(self) -> str:
        """
        Create an agent engine with custom memory topics.

        Subclasses must implement this method to define their specific memory topics
        and configuration for the agent engine.

        Returns:
            str: Agent engine ID
        """
        pass

    async def _init_agent(self) -> None:
        """
        Lazy initialization of agent resources.
        This now constructs the agent and its serializable auth.
        """
        if self.agent is None:
            # --- Environment setup ---
            httpx_client = httpx.AsyncClient(
                timeout=120,
                auth=GoogleAuth(),
            )
            httpx_client.headers["Content-Type"] = "application/json"
            # Create the actual agent
            self.agent = await get_orchestrator_agent(
                remote_agent_addresses=self.remote_agent_addresses,
                httpx_client=httpx_client,
            )

            # Configure memory and session services
            if self.agent_engine_id:
                # Use Vertex AI Memory Bank and Session Service for production
                my_memory_service = PersistentVertexAiMemoryBankService(
                    project=os.environ.get("PROJECT_ID"),
                    location=os.environ.get("LOCATION"),
                    agent_engine_id=self.agent_engine_id,
                )

                my_session_service = VertexAiSessionService(
                    project=os.environ.get("PROJECT_ID"),
                    location=os.environ.get("LOCATION"),
                    agent_engine_id=self.agent_engine_id,
                )
            else:
                # Use in-memory services for local testing
                my_memory_service = InMemoryMemoryService()
                my_session_service = InMemorySessionService()

            # The Runner orchestrates the agent execution
            # It manages the LLM calls, tool execution, and state
            self.runner = Runner(
                app_name=self.agent.name,
                agent=self.agent,
                artifact_service=InMemoryArtifactService(),
                session_service=my_session_service,
                memory_service=my_memory_service,
            )

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Process a user query and return the answer.

        This method is called by the A2A protocol handler when:
        1. A new message arrives (message/send)
        2. A streaming request is made (message/stream)
        """
        # Initialize agent on first call
        if self.agent is None:
            await self._init_agent()

        # Extract the user's question from the protocol message
        query = context.get_user_input()
        logging.info(f"Received query: {query}")

        # Create a TaskUpdater for managing task state
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        # Update task status through its lifecycle
        # submitted -> working -> completed/failed
        if not context.current_task:
            # New task - mark as submitted
            await updater.submit()

        # Mark task as working (processing)
        await updater.start_work()

        try:
            # Get or create a session for this conversation
            session = await self._get_or_create_session(context.context_id)
            logging.info(f"Using session: {session.id}")

            # Prepare the user message in ADK format
            content = types.Content(role=Role.user, parts=[types.Part(text=query)])

            # Run the agent asynchronously
            # This may involve multiple LLM calls and tool uses
            answer_sent = False
            async for event in self.runner.run_async(
                session_id=session.id,
                user_id="user",  # In production, use actual user ID
                new_message=content,
            ):
                # The agent may produce multiple events
                # We're interested in the final response
                if event.is_final_response() and not answer_sent:
                    # Extract the answer text from the response
                    answer = self._extract_answer(event)
                    logging.info(f" {answer}")

                    # Add the answer as an artifact
                    # Artifacts are the "outputs" or "results" of a task
                    # They're separate from status messages
                    await updater.add_artifact(
                        [TextPart(text=answer)],
                        name="answer",  # Name helps clients identify artifacts
                    )

                    # Mark task as completed successfully
                    await updater.complete()
                    answer_sent = True
                    # Don't break - continue consuming events to allow callbacks to execute

        except Exception as e:
            # Errors should never pass silently (Zen of Python)
            # Always inform the client when something goes wrong
            logging.error(f"Error during execution: {e!s}", exc_info=True)
            await updater.update_status(
                TaskState.failed, message=new_agent_text_message(f"Error: {e!s}")
            )
            # Re-raise for proper error handling up the stack
            raise

    async def _get_or_create_session(self, context_id: str):
        """Get existing session or create new one.

        Note: For Vertex AI Session Service, we create a new session each time
        because the A2A context_id format may not be compatible with Vertex AI's
        session resource name requirements. The session service will generate
        a valid session ID.
        """
        if isinstance(self.runner.session_service, InMemorySessionService):
            # For in-memory sessions, we can use the context_id directly
            session = await self.runner.session_service.get_session(
                app_name=self.runner.app_name,
                user_id="user",
                session_id=context_id,
            )

            if not session:
                logging.info(f"No session found for {context_id}, creating new one.")
                session = await self.runner.session_service.create_session(
                    app_name=self.runner.app_name,
                    user_id="user",
                    session_id=context_id,
                )
            else:
                logging.info(f"Found existing session {context_id}.")
        else:
            # For Vertex AI Session Service, create a new session without passing session_id
            # Let Vertex AI generate a valid session resource name
            logging.info(f"Creating new session for context {context_id}.")
            session = await self.runner.session_service.create_session(
                app_name=self.runner.app_name,
                user_id="user",
                # Don't pass session_id - let Vertex AI generate a valid one
            )

        return session

    def _extract_answer(self, event) -> str:
        """Extract text answer from agent response."""
        parts = event.content.parts
        text_parts = [part.text for part in parts if part.text]

        # Join all text parts with space
        return " ".join(text_parts) if text_parts else "No answer found."

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> NoReturn:
        """Handle task cancellation requests.

        For long-running agents, this would:
        1. Stop any ongoing processing
        2. Clean up resources
        3. Update task state to 'cancelled'
        """
        logging.warning(
            f"Cancellation requested for task {context.task_id}, but not supported."
        )
        # Inform client that cancellation isn't supported
        raise ServerError(error=UnsupportedOperationError())
