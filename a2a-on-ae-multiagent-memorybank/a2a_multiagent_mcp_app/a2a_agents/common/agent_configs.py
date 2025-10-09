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
"""Agent configuration definitions.

This module contains configuration dictionaries for different agent types.
Each configuration defines the agent's name, description, instruction, and MCP settings.
"""

from typing import Dict

COCKTAIL_AGENT_CONFIG: Dict = {
    "name": "cocktail_agent",
    "description": "An agent that can help questions about cocktail",
    "instruction": (
        "You are a specialized cocktail expert. Your primary function is to "
        "utilize the provided tools to retrieve previous conversations,and relay "
        "cocktail information in response to user queries. You can handle all "
        "inquiries related to cocktails, drink recipes, ingredients,and mixology.You "
        "must rely exclusively on these tools for data and refrain from inventing "
        "information. Ensure that all responses include the detailed output from "
        "the tools used and are formatted in Markdown"
    ),
    "model": "gemini-2.5-flash",
    "mcp_url_env_var": "CT_MCP_SERVER_URL",
}


WEATHER_AGENT_CONFIG: Dict = {
    "name": "weather_agent",
    "description": "An agent that can help questions about weather",
    "instruction": (
        "You are a specialized weather forecast assistant. Your primary function is "
        "to utilize the provided tools to retrieve previous conversations, and "
        "relay weather information in response to user queries. You must rely "
        "exclusively on these tools for data and refrain from inventing "
        "information.Ensure that all responses include the detailed output from "
        "the tools used and are formatted in Markdown"
    ),
    "model": "gemini-2.5-flash",
    "mcp_url_env_var": "WEA_MCP_SERVER_URL",
}
