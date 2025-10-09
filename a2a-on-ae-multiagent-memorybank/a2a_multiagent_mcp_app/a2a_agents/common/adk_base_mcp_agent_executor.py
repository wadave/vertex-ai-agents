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
import time
from abc import ABC, abstractmethod
from typing import Dict, NoReturn

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Role, TaskState, TextPart, UnsupportedOperationError
from a2a.utils import new_agent_text_message
from a2a.utils.errors import ServerError
from google import adk
from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory import VertexAiMemoryBankService
from google.adk.sessions import VertexAiSessionService
from google.genai import Client
from google.adk.tools.mcp_tool.mcp_toolset import (
    McpToolset,
    StreamableHTTPConnectionParams,
)
from google.auth import exceptions as google_auth_exceptions
from google.auth.transport import requests as google_auth_requests
from google.genai import types
from google.oauth2 import id_token as google_id_token


def get_gcp_auth_headers(audience: str) -> Dict[str, str]:
    """
    Fetches a Google Cloud OIDC token for a target audience using ADC.

    This simplified function relies entirely on Application Default Credentials (ADC)
    and the google-auth library. The library automatically handles checking for
    local credentials, service accounts, or querying the metadata server,
    making a manual fallback unnecessary.

    Args:
        audience: The full URL/URI of the target service (e.g., your Cloud Run URL),
                  which is used as the audience for the OIDC token.

    Returns:
        A dictionary with the "Authorization" header, or an empty
        dictionary if auth fails or is skipped (e.g., no credentials).
    """
    try:
        # This single call is the canonical way to get an OIDC token using ADC.
        # It automatically finds credentials (local, SA, or metadata server).
        auth_req = google_auth_requests.Request()
        token = google_id_token.fetch_id_token(auth_req, audience)

        logging.info("Successfully fetched OIDC token via google.auth.")
        return {"Authorization": f"Bearer {token}"}

    except google_auth_exceptions.DefaultCredentialsError:
        # This is expected in local environments without ADC setup.
        logging.warning(
            "No Google Cloud credentials found (DefaultCredentialsError). "
            "Skipping OIDC token fetch. This is normal for local dev."
        )

    except Exception as e:
        # Any other error means ADC was likely found but token minting failed
        # (e.g., IAM permissions, wrong audience, metadata server unreachable).
        logging.critical(
            f"An unexpected error occurred fetching OIDC token for audience "
            f"'{audience}': {e}",
            exc_info=True,
        )

    # Return an empty dict if any exception occurred
    return {}


class PersistentVertexAiMemoryBankService(VertexAiMemoryBankService):
    """
    Fixed version of VertexAiMemoryBankService that keeps the httpx client alive.

    The original implementation creates a new Client (and httpx client) for each request,
    which causes "Cannot send a request, as the client has been closed" errors in
    deployed Agent Engine environments.

    This subclass maintains a single persistent Client (and thus API client) for the
    lifetime of the service, preventing premature httpx client closure.
    """

    def __init__(
        self, project: str = None, location: str = None, agent_engine_id: str = None
    ):
        super().__init__(
            project=project, location=location, agent_engine_id=agent_engine_id
        )
        # Create and cache both the Client and API client once
        self._persistent_client = None
        self._persistent_api_client = None

    def _get_api_client(self):
        """Override to return a persistent API client instead of creating new ones."""
        if self._persistent_api_client is None:
            # Keep the Client object alive to prevent httpx client closure
            self._persistent_client = Client(
                vertexai=True, project=self._project, location=self._location
            )
            self._persistent_api_client = self._persistent_client._api_client
        return self._persistent_api_client


class TokenManager:
    """Manages OIDC token with automatic refresh on expiry."""

    def __init__(self, audience: str, refresh_buffer_seconds: int = 300):
        """
        Initialize TokenManager.

        Args:
            audience: The target service URL for OIDC token.
            refresh_buffer_seconds: Refresh token this many seconds before expiry.
                                   Default is 300 (5 minutes).
        """
        self.audience = audience
        self.refresh_buffer_seconds = refresh_buffer_seconds
        self._token = None
        self._expiry = None

    def get_headers(self) -> Dict[str, str]:
        """
        Get authorization headers with a fresh or cached token.

        Returns:
            Dictionary with Authorization header, or empty dict if auth unavailable.
        """
        current_time = time.time()

        # Refresh if token is None or about to expire
        if self._token is None or self._expiry is None or current_time >= self._expiry:
            headers = get_gcp_auth_headers(self.audience)
            auth_header = headers.get("Authorization")

            if auth_header:
                self._token = auth_header
                # ID tokens typically expire in 1 hour (3600 seconds)
                # Refresh 5 minutes (300 seconds) before expiry by default
                self._expiry = current_time + 3600 - self.refresh_buffer_seconds
                logging.info(
                    f"TokenManager: Refreshed token, next refresh at {self._expiry}"
                )
            else:
                # No token available
                self._token = None
                self._expiry = None

        # Return current token
        return {"Authorization": self._token} if self._token else {}


class AdkBaseMcpAgentExecutor(AgentExecutor, ABC):
    """Base Agent Executor that bridges A2A protocol with ADK agents using MCP tools.

    The executor handles:
    1. Protocol translation (A2A messages to/from agent format)
    2. Task lifecycle management (submitted -> working -> completed)
    3. Session management for multi-turn conversations
    4. Error handling and recovery
    5. MCP authentication and token management
    """

    def __init__(self, agent_engine_id: str = None) -> None:
        """Initialize with lazy loading pattern.

        Args:
            agent_engine_id: Optional agent engine ID. If not provided, creates a new one.
        """
        self.agent = None
        self.runner = None
        self.token_manager = None
        self.agent_engine_id = agent_engine_id

        self.project_id = os.environ.get("PROJECT_ID")
        self.location = os.environ.get("LOCATION")

        if self.agent_engine_id is None:
            self.agent_engine_id = self.get_agent_engine()

    @abstractmethod
    def get_agent_config(self) -> Dict:
        """
        Return agent configuration dictionary.

        Returns:
            Dict with keys: name, description, instruction, model, mcp_url_env_var
        """
        pass

    def get_agent_engine(self) -> str:
        """
        Create a basic agent engine with default memory configuration.

        Subclasses can override this method to customize memory topics
        and other agent engine settings specific to their use case.

        Returns:
            str: Agent engine ID
        """
        import vertexai

        client = vertexai.Client(
            project=self.project_id,
            location=self.location,
        )

        agent_engine = client.agent_engines.create(
            config={
                "context_spec": {
                    "memory_bank_config": {
                        "generation_config": {
                            "model": (
                                f"projects/{self.project_id}/locations/{self.location}/"
                                "publishers/google/models/gemini-2.5-flash"
                            )
                        }
                    }
                }
            }
        )
        agent_engine_id = agent_engine.api_resource.name.split("/")[-1]
        return agent_engine_id

    def _init_agent(self) -> None:
        """
        Lazy initialization of agent resources.
        This constructs the agent and its token manager using the config.
        """
        if self.agent is None:
            # Get agent configuration
            config = self.get_agent_config()

            # Use custom memory service that keeps httpx client alive
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

            # --- Environment setup ---
            mcp_url = os.getenv(config["mcp_url_env_var"])

            if not mcp_url:
                raise ValueError(
                    f"Required environment variable '{config['mcp_url_env_var']}' is not set. "
                    f"Please set it to the MCP server URL for {config['name']}."
                )

            # Initialize token manager for automatic token refresh
            self.token_manager = TokenManager(audience=mcp_url)
            mcp_auth_headers = self.token_manager.get_headers()

            mcp_server_params = StreamableHTTPConnectionParams(
                url=mcp_url,
                headers=mcp_auth_headers,
            )

            async def auto_save_session_to_memory_callback(callback_context):
                """
                Callback to save conversation session to Vertex AI Memory Bank.

                This callback is triggered after the agent completes processing.
                It extracts conversation events from the session and sends them to
                the Memory Bank service for processing. The service generates semantic
                memories that can be retrieved in future conversations.

                Memory topics are configured in the agent engine (see get_agent_engine method).
                Subclasses can override get_agent_engine to customize memory topics.
                """
                session = callback_context._invocation_context.session
                memory_service = callback_context._invocation_context.memory_service

                logging.info(
                    f"Saving session {session.id} to memory bank for "
                    f"user_id={session.user_id}"
                )

                try:
                    await memory_service.add_session_to_memory(session)
                    logging.info(
                        f"Memory generation completed for session {session.id}"
                    )
                except Exception as e:
                    logging.error(
                        f"Memory generation failed for session {session.id}: {e}",
                        exc_info=True,
                    )

            # Create the actual agent
            self.agent = LlmAgent(
                model=config.get("model", "gemini-2.5-flash"),
                name=config["name"],
                description=config["description"],
                instruction=config["instruction"],
                tools=[
                    McpToolset(
                        connection_params=mcp_server_params,
                    ),
                    adk.tools.preload_memory_tool.PreloadMemoryTool(),
                ],
                after_agent_callback=auto_save_session_to_memory_callback,
            )

            # The Runner orchestrates the agent execution
            # It manages the LLM calls, tool execution, and state
            self.runner = Runner(
                app_name=self.agent.name,
                agent=self.agent,
                # In-memory services for simplicity
                # In production, you might use persistent storage
                artifact_service=InMemoryArtifactService(),
                # session_service=InMemorySessionService(),
                # memory_service=InMemoryMemoryService(),
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
            self._init_agent()

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

        # Refresh MCP authentication headers before executing
        self._refresh_mcp_auth()

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

    def _refresh_mcp_auth(self) -> None:
        """Refresh MCP authentication headers using the token manager."""
        if self.token_manager is None:
            logging.warning("TokenManager not initialized, skipping auth refresh")
            return

        # Get fresh headers from token manager (will auto-refresh if expired)
        fresh_headers = self.token_manager.get_headers()

        # Update the toolset connection params (using private attribute)
        for tool in self.agent.tools:
            if isinstance(tool, McpToolset):
                # Access private attribute to update headers
                if hasattr(tool._connection_params, "headers"):
                    tool._connection_params.headers = fresh_headers
                    logging.debug("Refreshed MCP authentication headers")

    async def _get_or_create_session(self, context_id: str):
        """Get existing session or create new one."""
        # For Vertex AI Session Service, don't pass session_id to get_session
        # Instead, create a new session each time (stateless per A2A context)
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
