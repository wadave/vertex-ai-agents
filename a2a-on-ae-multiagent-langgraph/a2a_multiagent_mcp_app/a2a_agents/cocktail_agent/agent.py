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

from common.langgraph_base_mcp_agent import LanggraphBaseMCPAgent


class CocktailAgent(LanggraphBaseMCPAgent):
    """CocktailAgent - a specialized assistant for Cocktail information."""

    def __init__(self, mcp_tools: list[Any]):
        """Initialize the CocktailAgent with MCP tools."""
        super().__init__(mcp_tools)

    def get_system_instruction(self) -> str:
        """Return the system instruction for the Cocktail agent."""
        return """You are a specialized cocktail assistant. Your primary function is to utilize the provided tools to retrieve and relay cocktail information in response to user queries. You can handle all inquiries related to cocktails, drink recipes, ingredients, and mixology. You must rely exclusively on these tools for data and refrain from inventing information. Ensure that all responses include the detailed output from the tools used and are formatted in Markdown"""
