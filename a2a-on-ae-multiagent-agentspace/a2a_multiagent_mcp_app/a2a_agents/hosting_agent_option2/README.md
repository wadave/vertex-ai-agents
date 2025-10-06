# Set up Hosting Agent

This example shows how to create an ADK root agent on Agent Engine.

## Prerequisites

Before you begin, ensure you have the following:
 * You have deployed the remote agents, and have the agent urls WEA_AGENT_URL, and CT_AGENT_URL
 * You can follow `a2a-on-ae-multiagent` folder to dep`loy the remote agents if you have not deployed the remote agents.   

## Running the example

1. Create a `.env` file by copying the `.env.example` file:
   ```bash
   cp .env.example .env
   ```
   Then, update the `.env` file with your project-specific values.

2. Run the `deploy_adk_agent_on_agent_engine.ipynb` notebook to deploy the ADK root agent on Agent Engine.