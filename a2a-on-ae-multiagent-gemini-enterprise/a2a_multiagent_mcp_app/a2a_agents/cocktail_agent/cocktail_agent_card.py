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
# Helpers
import logging

# A2A
from a2a.types import AgentSkill

# Agent Engine
from vertexai.preview.reasoning_engines.templates.a2a import create_agent_card


logging.getLogger().setLevel(logging.INFO)

# Define a skill - a specific capability your agent offers
# Agents can have multiple skills for different tasks
cocktail_agent_skill = AgentSkill(
    # Unique identifier for this skill
    id="cocktail_cocktail",
    name="Search cocktail information",
    # Detailed description helps clients understand when to use this skill
    description="Helps with cocktail information search",
    # Tags for categorization and discovery
    # These help in agent marketplaces or registries
    tags=["cocktail", "drink", "recipe", "ingredients"],
    # Examples show clients what kinds of requests work well
    # This is especially helpful for LLM-based clients
    examples=[
        "List a random cocktail",
        "Find a cocktail with rum",
        "What are the ingredients for a Margarita?",
    ],
    # Optional: specify input/output modes
    # Default is text, but could include images, files, etc.
    input_modes=["text/plain"],
    output_modes=["text/plain"],
)

# Use the helper function to create a complete Agent Card
cocktail_agent_card = create_agent_card(
    agent_name="Cocktail Agent - ADK",
    description="A helpful assistant agent that can answer questions.",
    skills=[cocktail_agent_skill],
)
