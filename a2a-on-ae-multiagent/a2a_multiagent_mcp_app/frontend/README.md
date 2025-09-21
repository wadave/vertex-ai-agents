# A2A Multi-Agent on Agent Engine

> **⚠️ DISCLAIMER: THIS IS NOT AN OFFICIALLY SUPPORTED GOOGLE PRODUCT. THIS PROJECT IS INTENDED FOR DEMONSTRATION PURPOSES ONLY. IT IS NOT INTENDED FOR USE IN A PRODUCTION ENVIRONMENT.**

## **1. Run it locally**

```bash
uv run main.py
```

## **2. Deploy to Cloud Run**


In Cloud Shell, execute the following command:
```bash
# Define a name for your Cloud Run service
export SERVICE_NAME='a2a-frontend'

# Specify the Google Cloud region for deployment (ensure it supports required services)
export LOCATION='us-central1'

# Replace with your Google Cloud Project ID
export PROJECT_ID=''

export PROJECT_NUMBER=''
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
In Cloud Shell, execute the following command to autherize the Cloud Run service
```
   gcloud run services add-iam-policy-binding $SERVICE_NAME$                    \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"\
    --role="roles/run.invoker"           \
    --region=us-central1 --project=dw-genai-dev  

 ```