# Set Up MCP server using Cloud Run

## Setup and Deployment



In your Cloud Shell or local terminal (with gcloud CLI configured), set the following environment variables:

Note: The below parameters need to match with the values in .env file.

```bash
# Define a name for your Cloud Run service
export SERVICE_NAME='weather-remote-mcp-server'

# Specify the Google Cloud region for deployment (ensure it supports required services)
export LOCATION='us-central1'

# Replace with your Google Cloud Project ID
export PROJECT_ID='dw-genai-dev'

export PROJECT_NUMBER='496235138247'
```

In Cloud Shell, execute the following command:


```bash
gcloud run deploy $SERVICE_NAME \
  --source . \
  --region $LOCATION \
  --project $PROJECT_ID \
  --memory 4G \
  --no-allow-unauthenticated


```
```
gcloud run services proxy $SERVICE_NAME --region=us-central1
```

gcloud run services add-iam-policy-binding $SERVICE_NAME \
    --member="serviceAccount:496235138247-compute@developer.gserviceaccount.com" \
    --role="roles/run.invoker" \
    --region="us-central1"