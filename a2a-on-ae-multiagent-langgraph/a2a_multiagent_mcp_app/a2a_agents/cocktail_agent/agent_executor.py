import logging
import os

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

from agent import CocktailAgent


# Replace these as appropriate
PROJECT_ID = os.getenv("PROJECT_ID", "dw-genai-dev")
LOCATION = os.getenv("LOCATION", "us-central1")
STORAGE = os.getenv("BUCKET", "dw-genai-dev-bucket")

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
    staging_bucket="gs://" + STORAGE,
)


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)


class CocktailAgentExecutor(AgentExecutor):
    """Cocktail Agent Executor."""

    def __init__(self):
        """Initialize with lazy loading pattern."""
        self.agent = None
        self.mcp_client = None

    async def _init_agent(self) -> None:
        """
        Lazy initialization of agent resources.
        This now constructs the agent and its serializable auth.
        """
        if self.agent is None:
            # --- Environment setup ---
            url = os.environ.get(
                "MCP_SERVER_URL",
                "https://cocktail-remote-mcp-server-496235138247.us-central1.run.app/mcp/",
            )

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

            self.mcp_client = MultiServerMCPClient(
                {
                    "Cocktail": {
                        "url": url,
                        "transport": "streamable_http",
                        "httpx_client_factory": google_auth_client_factory,
                    }
                }
            )

            mcp_tools = await self.mcp_client.get_tools()
            logger.info(f"Retrieved {len(mcp_tools)} MCP tools")

            # Create the actual agent
            tool_count = len(mcp_tools) if mcp_tools else "no"
            logger.info(f"Initializing AgentExecutor with {tool_count} MCP tools.")

            self.agent = CocktailAgent(mcp_tools=mcp_tools)
            logger.info("CocktailAgent initialized successfully.")

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
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
        return False

    async def cancel(  # noqa: ARG002
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise ServerError(error=UnsupportedOperationError())
