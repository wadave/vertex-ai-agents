# Set up Cocktail Agent

This example shows how to create an A2A agent on Agent Engine.

## Running the example

1. Create a `.env` file by copying the `.env.example` file:

   ```bash
   cp .env.example .env
   ```

   Then, update the `.env` file with your project-specific values.

2. Run the `deploy_cocktail_a2a_on_agent_engine.ipynb` notebook to deploy the A2A agent to Agent Engine.

## Authentication

The agent uses Application Default Credentials (ADC) for authenticating to the MCP server.

**Setup:**

Ensure you have ADC configured:

```bash
gcloud auth application-default login
```

**Token Refresh:**

- Tokens are automatically refreshed 5 minutes before expiry
- Uses Google Cloud OIDC tokens with the MCP server URL as the audience
