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

import os

import httpx
from dotenv import load_dotenv

from common.langgraph_base_orchestrator_agent import (
    LanggraphBaseOrchestratorAgent,
)

load_dotenv()


class HostingAgent(LanggraphBaseOrchestratorAgent):
    """The host agent.

    This is the agent responsible for choosing which remote agents to send
    tasks to and coordinate their work.
    """

    def __init__(
        self,
        remote_agent_addresses: list[str],
        http_client: httpx.AsyncClient,
        task_callback=None,
    ):
        """Initialize the HostingAgent."""
        super().__init__(remote_agent_addresses, http_client, task_callback)

    def get_system_instruction(self, state: dict) -> str:
        """Generate the system prompt for the agent.

        Args:
            state: The graph state containing messages and other context

        Returns:
            str: The system prompt string
        """
        agents_list = self.agents if self.agents else "Loading agents..."
        current_agent = self.check_state(state).get("active_agent", "None")

        if self.debug_mode:
            import logging

            logger = logging.getLogger(__name__)
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
