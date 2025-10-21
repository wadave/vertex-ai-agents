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
"""Base Agent for MCP-based A2A agents."""

import logging
import os
from abc import ABC, abstractmethod
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


class LanggraphBaseMCPAgent(ABC):
    """Base class for MCP-based agents."""

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    FORMAT_INSTRUCTION = (
        "Set response status to input_required if the user needs to provide more "
        "information to complete the request. "
        "Set response status to error if there is an error while processing the request. "
        "Set response status to completed if the request is complete."
    )

    def __init__(self, mcp_tools: list[Any]):
        """Initialize the agent with MCP tools.

        Args:
            mcp_tools: List of MCP tools to provide to the agent
        """
        self.model = self._initialize_model()
        self.mcp_tools = mcp_tools
        if not self.mcp_tools:
            raise ValueError("No MCP tools provided to the Agent")

        self.graph = create_react_agent(
            self.model,
            tools=self.mcp_tools,
            checkpointer=memory,
            prompt=self.get_system_instruction(),
            response_format=(self.FORMAT_INSTRUCTION, ResponseFormat),
        )

    @abstractmethod
    def get_system_instruction(self) -> str:
        """Return the system instruction prompt for this agent.

        Returns:
            str: The system instruction
        """
        pass

    def _initialize_model(self):
        """Initialize the LLM model (Vertex AI or Google Generative AI).

        Returns:
            Initialized model instance

        Raises:
            ValueError: If model environment variable is not set
            Exception: If model initialization fails
        """
        try:
            model = os.getenv("GOOGLE_GENAI_MODEL", "gemini-2.5-flash")
            if not model:
                raise ValueError("GOOGLE_GENAI_MODEL environment variable is not set")

            if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "TRUE").lower() in ["true", "1"]:
                # Using Vertex AI
                logger.info("ChatVertexAI model initialized successfully.")
                return ChatVertexAI(model=model)
            else:
                # Using Google Generative AI
                logger.info("ChatGoogleGenerativeAI model initialized successfully.")
                return ChatGoogleGenerativeAI(model=model)

        except Exception as e:
            logger.error(
                f"Failed to initialize model: {e}",
                exc_info=True,
            )
            raise

    async def stream(
        self, query: str, context_id: str
    ) -> AsyncIterable[dict[str, Any]]:
        """Stream responses from the agent.

        Args:
            query: User query
            context_id: Context ID for the conversation thread

        Yields:
            dict: Response dictionaries with status and content
        """
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
                    "content": self.get_tool_lookup_message(),
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
                    "content": self.get_tool_processing_message(),
                }

        yield self.get_agent_response(config)

    def get_tool_lookup_message(self) -> str:
        """Return the message to display when looking up information.

        Returns:
            str: The lookup message
        """
        return "Looking up the information..."

    def get_tool_processing_message(self) -> str:
        """Return the message to display when processing information.

        Returns:
            str: The processing message
        """
        return "Processing the information.."

    def get_agent_response(self, config: dict) -> dict[str, Any]:
        """Get the final agent response based on the structured response.

        Args:
            config: Configuration dict with thread_id

        Returns:
            dict: Response dictionary with status and content
        """
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
