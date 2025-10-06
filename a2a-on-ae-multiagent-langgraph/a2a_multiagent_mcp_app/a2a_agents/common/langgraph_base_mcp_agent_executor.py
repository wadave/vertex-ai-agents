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
"""Base Agent Executor for MCP-based A2A agents."""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import httpx
import vertexai
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import (
    new_agent_text_message,
    new_task,
)
from a2a.utils.errors import ServerError
from google.auth.transport.requests import Request as AuthRequest
from google.oauth2.id_token import fetch_id_token
from langchain_mcp_adapters.client import MultiServerMCPClient


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)


class LanggraphBaseMCPAgentExecutor(AgentExecutor, ABC):
    """Base class for MCP-based Agent Executors."""

    def __init__(self):
        """Initialize with lazy loading pattern."""
        self.agent = None
        self.mcp_client = None
        self._init_vertexai()

    def _init_vertexai(self) -> None:
        """Initialize Vertex AI with project configuration."""
        project_id = os.getenv("PROJECT_ID", "dw-genai-dev")
        location = os.getenv("LOCATION", "us-central1")
        storage = os.getenv("BUCKET", "dw-genai-dev-bucket")

        vertexai.init(
            project=project_id,
            location=location,
            staging_bucket="gs://" + storage,
        )

    @abstractmethod
    def get_mcp_server_url(self) -> str:
        """Return the MCP server URL for this agent.

        Returns:
            str: The MCP server URL
        """
        pass

    @abstractmethod
    def get_mcp_server_name(self) -> str:
        """Return the MCP server name for this agent.

        Returns:
            str: The MCP server name
        """
        pass

    @abstractmethod
    def create_agent(self, mcp_tools: list[Any]) -> Any:
        """Create and return the agent instance.

        Args:
            mcp_tools: List of MCP tools to provide to the agent

        Returns:
            Agent instance
        """
        pass

    def _create_google_auth_client_factory(self, url: str):
        """Create factory that creates httpx.AsyncClient with Google Auth.

        Args:
            url: The MCP server URL for authentication

        Returns:
            Factory function for creating authenticated httpx clients
        """

        def google_auth_client_factory(
            headers=None,
            timeout=None,
            auth=None,  # noqa: ARG001
        ):
            """Factory that creates httpx.AsyncClient with Google Auth."""
            auth_request = AuthRequest()
            id_token = fetch_id_token(auth_request, url)

            # Merge custom headers with auth headers
            client_headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {id_token}",
            }
            if headers:
                client_headers.update(headers)

            return httpx.AsyncClient(
                headers=client_headers,
                timeout=timeout if timeout is not None else 120,
            )

        return google_auth_client_factory

    async def _init_agent(self) -> None:
        """Lazy initialization of agent resources."""
        if self.agent is None:
            url = os.environ.get("MCP_SERVER_URL", self.get_mcp_server_url())
            server_name = self.get_mcp_server_name()

            self.mcp_client = MultiServerMCPClient(
                {
                    server_name: {
                        "url": url,
                        "transport": "streamable_http",
                        "httpx_client_factory": self._create_google_auth_client_factory(
                            url
                        ),
                    }
                }
            )

            mcp_tools = await self.mcp_client.get_tools()
            logger.info(f"Retrieved {len(mcp_tools)} MCP tools")

            tool_count = len(mcp_tools) if mcp_tools else "no"
            logger.info(f"Initializing AgentExecutor with {tool_count} MCP tools.")

            self.agent = self.create_agent(mcp_tools)
            logger.info(f"{self.__class__.__name__} initialized successfully.")

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute the agent with the given request context.

        Args:
            context: Request context containing user input and task information
            event_queue: Event queue for publishing task updates
        """
        if self.agent is None:
            await self._init_agent()

        error = self._validate_request(context)
        if error:
            raise ServerError(error=InvalidParamsError())

        query = context.get_user_input()
        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        try:
            async for item in self.agent.stream(query, task.context_id):
                is_task_complete = item["is_task_complete"]
                require_user_input = item["require_user_input"]

                if not is_task_complete and not require_user_input:
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(
                            item["content"],
                            task.context_id,
                            task.id,
                        ),
                    )
                elif require_user_input:
                    await updater.update_status(
                        TaskState.input_required,
                        new_agent_text_message(
                            item["content"],
                            task.context_id,
                            task.id,
                        ),
                        final=True,
                    )
                    break
                else:
                    await updater.add_artifact(
                        [Part(root=TextPart(text=item["content"]))],
                        name="conversion_result",
                    )
                    await updater.complete()
                    break

        except Exception as e:
            logger.error(f"An error occurred while streaming the response: {e}")
            raise ServerError(error=InternalError()) from e

    def _validate_request(self, context: RequestContext) -> bool:  # noqa: ARG002
        """Validate the request. Override in subclass if needed.

        Args:
            context: Request context to validate

        Returns:
            bool: True if validation failed, False otherwise
        """
        return False

    async def cancel(  # noqa: ARG002
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Cancel operation is not supported.

        Args:
            context: Request context
            event_queue: Event queue
        """
        raise ServerError(error=UnsupportedOperationError())
