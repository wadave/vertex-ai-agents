"""
Main function to run Root Agent.
"""

import logging
from typing import Dict

# Imports from the google-auth library
from google.auth import exceptions as google_auth_exceptions
from google.auth.transport import requests as google_auth_requests
from google.oauth2 import id_token as google_id_token
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import (
    MCPToolset,
    StdioServerParameters,
    StreamableHTTPConnectionParams,
)
import os

# Use the google-auth helpers when available to fetch ID tokens. Keep a
# metadata-server fallback for GCP compute environments.
from .prompts import (
    ROOT_AGENT_INSTRUCTION,
    COCKTAIL_AGENT_INSTRUCTION,
    BOOKING_AGENT_INSTRUCTION,
)

# Note: older examples/imports referenced `StreamableHttpServerParameters` from the
# third-party `mcp` package. The ADK now exposes a compatible wrapper type
# `StreamableHTTPConnectionParams` in `google.adk.tools.mcp_tool.mcp_toolset`.
# Importing from `mcp` directly can fail when that symbol isn't exported by the
# installed `mcp` package, which causes import-time failures when the ADK web
# server tries to load this app. Use the ADK-provided type for compatibility.

load_dotenv()

cocktail_server_name = os.getenv("COCKTAIL_REMOTE_MCP_SERVER_NAME")
weather_server_name = os.getenv("WEATHER_REMOTE_MCP_SERVER_NAME")
project_number = os.getenv("PROJECT_NUMBER")
region = os.getenv("GOOGLE_CLOUD_LOCATION", 'us-central1')


# --- Agent Application Definition ---

def get_gcp_auth_headers(audience: str) -> Dict[str, str]:
    """
    Fetches a Google Cloud OIDC token for a target audience using ADC.

    This simplified function relies entirely on Application Default Credentials (ADC)
    and the google-auth library. The library automatically handles checking for
    local credentials, service accounts, or querying the metadata server,
    making a manual fallback unnecessary.

    Args:
        audience: The full URL/URI of the target service (e.g., your Cloud Run URL),
                  which is used as the audience for the OIDC token.

    Returns:
        A dictionary with the "Authorization" header, or an empty
        dictionary if auth fails or is skipped (e.g., no credentials).
    """
    try:
        # This single call is the canonical way to get an OIDC token using ADC.
        # It automatically finds credentials (local, SA, or metadata server).
        auth_req = google_auth_requests.Request()
        token = google_id_token.fetch_id_token(auth_req, audience)

        logging.info("Successfully fetched OIDC token via google.auth.")
        return {"Authorization": f"Bearer {token}"}

    except google_auth_exceptions.DefaultCredentialsError:
        # This is expected in local environments without ADC setup.
        logging.warning(
            "No Google Cloud credentials found (DefaultCredentialsError). "
            "Skipping OIDC token fetch. This is normal for local dev."
        )
        
    except Exception as e:
        # Any other error means ADC was likely found but token minting failed
        # (e.g., IAM permissions, wrong audience, metadata server unreachable).
        logging.critical(
            f"An unexpected error occurred fetching OIDC token for audience '{audience}': {e}",
            exc_info=True
        )

    # Return an empty dict if any exception occurred
    return {}


wea_url = f"https://{weather_server_name}-{project_number}.{region}.run.app/mcp/"
ct_url = f"https://{cocktail_server_name}-{project_number}.{region}.run.app/mcp/"

wea_auth_headers = get_gcp_auth_headers(wea_url)
ct_auth_headers = get_gcp_auth_headers(ct_url)

weather_server_params = StreamableHTTPConnectionParams(
    url=wea_url,
    headers=wea_auth_headers,
)

ct_server_params = StreamableHTTPConnectionParams(
    url=ct_url,
    headers=ct_auth_headers,
)

bnb_server_params = StdioServerParameters(
    command="npx", args=["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"]
)


MODEL_ID = "gemini-2.5-flash"


# --- Agent Creation ---
def create_agent() -> LlmAgent:
    """
    Creates the root LlmAgent and its sub-agents using pre-loaded MCP tools.

    Args:
        loaded_mcp_tools: A dictionary of tools, typically populated at application
                        startup, where keys are toolset identifiers (e.g., "bnb",
                        "weather", "ct") and values are the corresponding tools.

    Returns:
        An LlmAgent instance representing the root agent, configured with sub-agents.
    """
    booking_agent = LlmAgent(
        model=MODEL_ID,
        name="booking_assistant",
        instruction=BOOKING_AGENT_INSTRUCTION,
        tools=[
            MCPToolset(connection_params=bnb_server_params),
            MCPToolset(connection_params=weather_server_params),
        ],
    )

    cocktail_agent = LlmAgent(
        model=MODEL_ID,
        name="cocktail_assistant",
        instruction=COCKTAIL_AGENT_INSTRUCTION,
        tools=[MCPToolset(connection_params=ct_server_params)],
    )

    root_agent = LlmAgent(
        model=MODEL_ID,
        name="ai_assistant",
        instruction=ROOT_AGENT_INSTRUCTION,
        sub_agents=[cocktail_agent, booking_agent],
    )
    return root_agent


root_agent = create_agent()
