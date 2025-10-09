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
from common.adk_orchestrator_agent_executor import AdkOrchestratorAgentExecutor
import logging
import os
from dotenv import load_dotenv

# Set logging
logging.getLogger().setLevel(logging.INFO)
load_dotenv()


class HostingAgentExecutor(AdkOrchestratorAgentExecutor):
    """Agent Executor that wraps OrchestratorAgentExecutor with environment-based
    configuration.

    This class provides backward compatibility by reading remote agent addresses
    from environment variables and delegating to OrchestratorAgentExecutor.
    """

    def __init__(self, agent_engine_id: str = None) -> None:
        """Initialize with remote agent addresses from environment variables.

        Args:
            agent_engine_id: Optional agent engine ID for memory bank. If not
              provided, creates a new one.
        """
        remote_agent_addresses = [
            os.getenv("CT_AGENT_URL", "http://localhost:10002"),
            os.getenv("WEA_AGENT_URL", "http://localhost:10001"),
        ]
        super().__init__(
            remote_agent_addresses=remote_agent_addresses,
            agent_engine_id=agent_engine_id,
        )

    def get_agent_engine(self) -> str:
        """
        Create an agent engine with orchestrator-specific memory topics.

        Configures the following memory topics:
        - task_delegation: Custom topic for tracking delegated tasks
        - agent_routing: Custom topic for agent routing decisions
        - USER_PERSONAL_INFO: User personal information
        - USER_PREFERENCES: User preferences
        - KEY_CONVERSATION_DETAILS: Important conversation details
        - EXPLICIT_INSTRUCTIONS: User instructions

        Returns:
            str: Agent engine ID
        """
        import vertexai

        project_id = os.environ.get("PROJECT_ID")
        location = os.environ.get("LOCATION")

        client = vertexai.Client(
            project=project_id,
            location=location,
        )

        orchestrator_memory_config = {
            # "generate_memories_examples": [example],
            "memory_topics": [
                {
                    "custom_memory_topic": {
                        "label": "task_delegation",
                        "description": "Information about tasks delegated to specialized agents",
                    }
                },
                {
                    "custom_memory_topic": {
                        "label": "agent_routing",
                        "description": "Information about which agents were selected for which types of queries",
                    }
                },
                {
                    "custom_memory_topic": {
                        "label": "cocktail_id",
                        "description": "cocktail id retrieved from remote agents",
                    }
                },
                {
                    "custom_memory_topic": {
                        "label": "cocktail_recipe",
                        "description": "cocktail recipe from remote agent",
                    }
                },
                {
                    "custom_memory_topic": {
                        "label": "cocktail_ingredients",
                        "description": "cocktail ingredients from remote agent",
                    }
                },
                {
                    "custom_memory_topic": {
                        "label": "location",
                        "description": "city and state mentioned in the conversation",
                    }
                },
                {
                    "custom_memory_topic": {
                        "label": "weather_forecast",
                        "description": "weather forecast details from remote agent",
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
                            "model": f"projects/{project_id}/locations/{location}/publishers/google/models/gemini-2.5-flash"
                        },
                        "customization_configs": [orchestrator_memory_config],
                    }
                }
            }
        )
        agent_engine_id = agent_engine.api_resource.name.split("/")[-1]
        return agent_engine_id
