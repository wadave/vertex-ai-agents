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

from common.agent_configs import COCKTAIL_AGENT_CONFIG
from common.adk_base_mcp_agent_executor import AdkBaseMcpAgentExecutor

# Set logging
logging.getLogger().setLevel(logging.INFO)
load_dotenv()


class CocktailAgentExecutor(AdkBaseMcpAgentExecutor):
    """Agent Executor for cocktail-related queries using MCP tools."""

    def get_agent_config(self) -> Dict:
        """Return cocktail agent configuration."""
        return COCKTAIL_AGENT_CONFIG

    def get_agent_engine(self) -> str:
        """
        Create an agent engine with cocktail-specific memory topics.

        Configures the following memory topics:
        - cocktail_id: Custom topic for cocktail identifiers
        - USER_PERSONAL_INFO: User personal information
        - USER_PREFERENCES: User preferences (e.g., favorite cocktails)
        - KEY_CONVERSATION_DETAILS: Important conversation details
        - EXPLICIT_INSTRUCTIONS: User instructions

        Returns:
            str: Agent engine ID
        """
        import vertexai

        client = vertexai.Client(
            project=self.project_id,
            location=self.location,
        )

        example = {
            "conversationSource": {
                "events": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "text": (
                                        "Here is the recipe for the Margarita cocktail:\n\n"
                                        "Margarita\n"
                                        "ID: 11007\n"
                                        "Category: Ordinary Drink\n"
                                        "Glass: Cocktail glass\n"
                                        "Alcoholic: Alcoholic\n"
                                        "Instructions: Rub the rim of the glass with the lime "
                                        "slice to make the salt stick to it. Take care to "
                                        "moisten only the outer rim and sprinkle the salt on "
                                        "it. The salt should be applied to the outside edge of "
                                        "the glass. Shake the tequila, Cointreau, and lime "
                                        "juice with ice, then strain into the prepared glass."
                                    )
                                }
                            ],
                        }
                    },
                    {
                        "content": {
                            "role": "user",
                            "parts": [
                                {"text": "What are the recipes for a Margarita?"}
                            ],
                        }
                    },
                ]
            },
            "generatedMemories": [
                {
                    "fact": (
                        "Margarita\n"
                        "ID: 11007\n"
                        "Category: Ordinary Drink\n"
                        "Glass: Cocktail glass\n"
                        "Alcoholic: Alcoholic\n"
                        "Instructions: Rub the rim of the glass with the lime slice to "
                        "make the salt stick to it. Take care to moisten only the "
                        "outer rim and sprinkle the salt on it. The salt should be "
                        "applied to the outside edge of the glass. Shake the tequila, "
                        "Cointreau, and lime juice with ice, then strain into the "
                        "prepared glass.."
                    )
                },
                {"fact": "The user asked for ingredients for a Margarita."},
            ],
        }

        user_preferences_config = {
            "generate_memories_examples": [example],
            "memory_topics": [
                {
                    "custom_memory_topic": {
                        "label": "cocktail_id",
                        "description": "cocktail id retrieved from MCP server",
                    }
                },
                {
                    "custom_memory_topic": {
                        "label": "cocktail_recipe",
                        "description": "cocktail recipe from MCP server",
                    }
                },
                {
                    "custom_memory_topic": {
                        "label": "cocktail_ingredients",
                        "description": "cocktail ingredients from MCP server",
                    }
                },
                {"managed_memory_topic": {"managed_topic_enum": "USER_PERSONAL_INFO"}},
                {"managed_memory_topic": {"managed_topic_enum": "USER_PREFERENCES"}},
                {
                    "managed_memory_topic": {
                        "managed_topic_enum": "KEY_CONVERSATION_DETAILS"
                    }
                },
                {
                    "managed_memory_topic": {
                        "managed_topic_enum": "EXPLICIT_INSTRUCTIONS"
                    }
                },
            ],
        }

        agent_engine = client.agent_engines.create(
            config={
                "context_spec": {
                    "memory_bank_config": {
                        "generation_config": {
                            "model": (
                                f"projects/{self.project_id}/locations/{self.location}/"
                                "publishers/google/models/gemini-2.5-flash"
                            )
                        },
                        "customization_configs": [user_preferences_config],
                    }
                }
            }
        )
        agent_engine_id = agent_engine.api_resource.name.split("/")[-1]
        return agent_engine_id
