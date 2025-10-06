# Multi-Agent System using LangGraph

This project implements a multi-agent architecture using Python and the LangGraph library. The system is composed of a central orchestrator agent and several specialized "tool" agents that perform specific tasks.

## Overview

The core of this project is the `hosting_agent`, which acts as an orchestrator. It receives incoming requests and delegates them to the appropriate specialized agent based on the nature of the request.

The specialized agents include:
- **Cocktail Agent**: Provides information and recipes for cocktails.
- **Weather Agent**: Fetches current weather information for a given location.

## Project Structure

- `hosting_agent/`: Contains the main orchestrator agent that routes requests to other agents.
- `cocktail_agent/`: A specialized agent for providing cocktail recipes.
- `weather_agent/`: A specialized agent for fetching weather forecasts.
- `common/`: Contains shared base classes and utilities for the LangGraph agents and executors, promoting code reuse and a consistent structure.
- `deploy_*.ipynb`: Jupyter notebooks providing a step-by-step guide to deploying the host, cocktail, and weather agents.
- `.env.example`: An example file showing the required environment variables for the project.

## Setup

1.  **Install Dependencies**: This project requires Python dependencies. :
    ```bash
    uv sync
    ```

2.  **Configure Environment**: Copy the `.env.example` file to a new file named `.env` in the root of this folder.
    ```bash
    cp .env.example .env
    ```
    Edit the `.env` file to include the necessary credentials and configuration values for your Google Cloud project and any other required APIs.

## Deployment and Usage

The primary method for deploying and running the agents is through the provided Jupyter Notebooks:

- `deploy_langgraph_host_agent.ipynb`
- `deploy_cocktail_langgraph_agent.ipynb`
- `deploy_weather_langgraph_agent.ipynb`

Open these notebooks and execute the cells in order to deploy each agent as a service on Agent Enigne. The host agent will then be ableto communicate with the other deployed agents.
