import logging
import os

from collections.abc import AsyncIterable
from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

memory = MemorySaver()


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str


class WeatherAgent:
    """WeatherAgent - a specialized assistant for weather forecasts."""

    SYSTEM_INSTRUCTION = (
        "You are a specialized weather forecast assistant. Your primary function is "
        "to utilize the provided tools to retrieve and relay weather information in "
        "response to user queries. You must rely exclusively on these tools for data "
        "and refrain from inventing information. Ensure that all responses include "
        "the detailed output from the tools used and are formatted in Markdown"
    )

    FORMAT_INSTRUCTION = (
        "Set response status to input_required if the user needs to provide more "
        "information to complete the request. "
        "Set response status to error if there is an error while processing the request. "
        "Set response status to completed if the request is complete."
    )

    def __init__(self, mcp_tools: list[Any]):
        try:
            model = os.getenv("GOOGLE_GENAI_MODEL", "gemini-2.5-flash")
            if not model:
                raise ValueError("GOOGLE_GENAI_MODEL environment variable is not set")

            if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "TRUE").lower() in ["true", "1"]:
                # Using Vertex AI
                logger.info("ChatVertexAI model initialized successfully.")
                self.model = ChatVertexAI(model=model)
            else:
                # Using Google Generative AI
                self.model = ChatGoogleGenerativeAI(model=model)
                logger.info("ChatGoogleGenerativeAI model initialized successfully.")

        except Exception as e:
            logger.error(
                f"Failed to initialize model: {e}",
                exc_info=True,
            )
            raise

        self.mcp_tools = mcp_tools
        if not self.mcp_tools:
            raise ValueError("No MCP tools provided to the Agent")

        self.graph = create_react_agent(
            self.model,
            tools=self.mcp_tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=(self.FORMAT_INSTRUCTION, ResponseFormat),
        )

    async def stream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        inputs = {"messages": [("user", query)]}
        config = {"configurable": {"thread_id": context_id}}

        async for item in self.graph.astream(inputs, config, stream_mode="values"):
            message = item["messages"][-1]
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                logger.info(f"Tool calls: {message.tool_calls}")
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Looking up the information...",
                }
            elif isinstance(message, ToolMessage):
                logger.info(f"Tool message content: {message.content}")
                status = message.status if hasattr(message, "status") else "N/A"
                logger.info(f"Tool message status: {status}")
                if hasattr(message, "artifact") and message.artifact:
                    logger.error(f"Tool error: {message.artifact}")
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Processing the information..",
                }

        yield self.get_agent_response(config)

    def get_agent_response(self, config):
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get("structured_response")
        if structured_response and isinstance(structured_response, ResponseFormat):
            if structured_response.status == "input_required":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message,
                }
            if structured_response.status == "error":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message,
                }
            if structured_response.status == "completed":
                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": structured_response.message,
                }

        return {
            "is_task_complete": False,
            "require_user_input": True,
            "content": (
                "We are unable to process your request at the moment. Please try again."
            ),
        }

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]
