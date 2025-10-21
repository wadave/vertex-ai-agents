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
from typing import Any

from common.langgraph_base_mcp_agent_executor import (
    LanggraphBaseMCPAgentExecutor,
)

from weather_agent.agent import WeatherAgent


class WeatherAgentExecutor(LanggraphBaseMCPAgentExecutor):
    """Weather Agent Executor."""

    def get_mcp_server_url(self) -> str:
        """Return the MCP server URL for Weather agent."""
        return "https://weather-remote-mcp-server-496235138247.us-central1.run.app/mcp/"

    def get_mcp_server_name(self) -> str:
        """Return the MCP server name for Weather agent."""
        return "weather"

    def create_agent(self, mcp_tools: list[Any]) -> WeatherAgent:
        """Create and return the WeatherAgent instance."""
        return WeatherAgent(mcp_tools=mcp_tools)
