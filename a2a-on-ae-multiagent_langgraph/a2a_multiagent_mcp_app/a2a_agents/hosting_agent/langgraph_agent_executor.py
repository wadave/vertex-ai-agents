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

import logging
import os

import httpx
import vertexai
from dotenv import load_dotenv
from google.auth import default
from google.auth.transport.requests import Request as AuthRequest
from langgraph.errors import GraphRecursionError

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError
from langgraph_agent import get_root_agent

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

# Vertex AI configuration
PROJECT_ID = os.getenv('PROJECT_ID', "dw-genai-dev")
LOCATION = os.getenv('LOCATION', "us-central1")
STORAGE = os.getenv('BUCKET', "dw-genai-dev-bucket")

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
    staging_bucket=f"gs://{STORAGE}",
)

# Debug mode from environment
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"


class GoogleAuth(httpx.Auth):
    """A custom httpx Auth class for Google Cloud authentication."""

    def __init__(self):
        self.credentials, self.project = default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self.auth_request = AuthRequest()

    def auth_flow(self, request):
        """Authenticate the request with Google Cloud credentials."""
        if not self.credentials.valid:
            logger.info("Refreshing expired Google Cloud credentials")
            self.credentials.refresh(self.auth_request)

        request.headers["Authorization"] = f"Bearer {self.credentials.token}"
        yield request


class HostingAgentExecutor(AgentExecutor):
    """Agent Executor that bridges A2A protocol with LangGraph agent.

    The executor handles:
    1. Protocol translation (A2A messages to/from LangGraph format)
    2. Task lifecycle management (submitted -> working -> completed)
    3. Session management for multi-turn conversations via checkpointing
    4. Error handling and recovery, including recursion limit handling
    """

    def __init__(self) -> None:
        """Initialize with lazy loading pattern."""
        self.agent = None

    async def _init_agent(self) -> None:
        """Lazy initialization of agent resources.

        Creates an HTTP client with Google Cloud authentication and
        initializes the LangGraph agent with all remote agent cards loaded.
        """
        if self.agent is None:
            httpx_client = httpx.AsyncClient(
                timeout=120,
                auth=GoogleAuth(),
            )
            httpx_client.headers["Content-Type"] = "application/json"
            self.agent = await get_root_agent(httpx_client)

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
        logger.info(f"Received query: {query}")

        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)

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
            # Use astream() for async tool support
            # Build the config with the thread_id for conversation context
            config = {
                "configurable": {"thread_id": task.context_id},
                "recursion_limit": 10,  # Allow up to 10 iterations for multi-turn conversations
            }

            final_response = None
            needs_input = False

            # Stream through the graph execution
            iteration_count = 0
            try:
                async for chunk in self.agent.astream({"messages": [("user", query)]}, config, stream_mode="values"):
                    iteration_count += 1

                    if DEBUG_MODE:
                        logger.debug(f"Iteration {iteration_count}")

                    # Each chunk contains the full state with messages
                    if "messages" in chunk:
                        messages = chunk["messages"]
                        if messages:
                            last_message = messages[-1]
                            message_type = type(last_message).__name__

                            # Check if this is an AIMessage without tool_calls (final response)
                            if message_type == 'AIMessage' and (not hasattr(last_message, 'tool_calls') or not last_message.tool_calls):
                                if hasattr(last_message, 'content') and last_message.content:
                                    final_response = str(last_message.content)
                                    if DEBUG_MODE:
                                        logger.debug(f"AI final response: {final_response[:100]}")

                            # Check if it's an AI message
                            if hasattr(last_message, "content") and last_message.content:
                                content = last_message.content

                                # Check if content is structured (ResponseFormat from Pydantic)
                                # The response_format may return a Pydantic model instance
                                if hasattr(content, "model_dump"):
                                    # It's a Pydantic model - convert to dict
                                    content_dict = content.model_dump()
                                    final_response = content_dict.get("message", str(content_dict))
                                    status = content_dict.get("status", "completed")
                                    needs_input = (status == "input_required")
                                elif isinstance(content, dict):
                                    # Already a dict with status field
                                    final_response = content.get("message", str(content))
                                    status = content.get("status", "completed")
                                    needs_input = (status == "input_required")
                                else:
                                    # Plain text response
                                    final_response = content

                                # Check state for input requirement
                                if "needs_user_input" in chunk and chunk["needs_user_input"]:
                                    needs_input = True

            except GraphRecursionError as e:
                # Handle recursion limit gracefully
                logger.warning(f"Recursion limit reached after {iteration_count} iterations: {e}")
                final_response = "I've reached the maximum number of steps for this task. The query may be too complex or require a different approach. Please try rephrasing your request or breaking it into smaller steps."
                needs_input = False

            # Process the final result
            if needs_input:
                await updater.update_status(
                    TaskState.input_required,
                    new_agent_text_message(
                        final_response or "Additional input required",
                        task.context_id,
                        task.id,
                    ),
                    final=True,
                )
            elif final_response:
                # Extract text response from the final message
                response_text = final_response if isinstance(final_response, str) else str(final_response)

                await updater.add_artifact(
                    [Part(root=TextPart(text=response_text))],
                    name='response',
                )
                await updater.complete()
            else:
                raise ValueError("No response received from agent")

        except Exception as e:
            logger.error(f'An error occurred while streaming the response: {e}')
            raise ServerError(error=InternalError()) from e

    def _validate_request(self, context: RequestContext) -> bool:
        """Validate incoming request (not implemented)."""
        return False

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel execution (not supported)."""
        raise ServerError(error=UnsupportedOperationError())
