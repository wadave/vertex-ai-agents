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
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    AuthorizationCodeOAuthFlow,
    OAuth2SecurityScheme,
    OAuthFlows,
    SecurityScheme,
)

# Agent Engine
from vertexai.preview.reasoning_engines.templates.a2a import create_agent_card


logging.getLogger().setLevel(logging.INFO)

# Define a skill - a specific capability your agent offers
# Agents can have multiple skills for different tasks

OAUTH_SCHEME_NAME = "GoogleOAuth"

oauth_scheme = OAuth2SecurityScheme(
    type="oauth2",
    description="OAuth2 for Google Calendar API",
    flows=OAuthFlows(
        authorization_code=AuthorizationCodeOAuthFlow(
            authorization_url="https://accounts.google.com/o/oauth2/auth",
            token_url="https://oauth2.googleapis.com/token",
            scopes={
                "https://www.googleapis.com/auth/cloud-platform": "Full access to Google Cloud Platform services",
                "openid": "Authenticate user identity",
                "email": "Access user's email address",
                "profile": "Access user's basic profile information",
            },
        )
    ),
)

hosting_agent_skill = AgentSkill(
    # Unique identifier for this skill
    id="hosting_agent",
    name="Search hosting agent",
    # Detailed description helps clients understand when to use this skill
    description="Helps with weather in city, or states, and cocktails",
    tags=["host_agent"],
    examples=["weather in LA, CA", "List a random cocktail", "What is a margarita?"],
    # Optional: specify input/output modes
    # Default is text, but could include images, files, etc.
    input_modes=["text/plain"],
    output_modes=["text/plain"],
)

agent_card = AgentCard(
    name="Hosting Agent - Gemini Enterprise",
    description="A helpful assistant agent that can answer questions.",
    skills=[hosting_agent_skill],
    url="http://localhost:9999/",
    preferred_transport="HTTP+JSON",
    version="1.0.0",
    default_input_modes=["text"],
    default_output_modes=["text"],
    capabilities=AgentCapabilities(streaming=False),
    # preferred_transport=TransportProtocol.http_json,  # Http Only.
    supports_authenticated_extended_card=True,
    security_schemes={OAUTH_SCHEME_NAME: SecurityScheme(root=oauth_scheme)},
    # Declare that this scheme is required to use the agent's skills
    security=[
        {
            OAUTH_SCHEME_NAME: [
                "https://www.googleapis.com/auth/cloud-platform",
                "openid",
                "email",
                "profile",
            ]
        }
    ],
)
