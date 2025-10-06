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
from typing import Dict
from dotenv import load_dotenv

from common.agent_configs import WEATHER_AGENT_CONFIG
from common.adk_base_mcp_agent_executor import AdkBaseMcpAgentExecutor

# Set logging
logging.getLogger().setLevel(logging.INFO)
load_dotenv()


class WeatherAgentExecutor(AdkBaseMcpAgentExecutor):
    """Agent Executor for weather-related queries using MCP tools."""

    def get_agent_config(self) -> Dict:
        """Return weather agent configuration."""
        return WEATHER_AGENT_CONFIG
