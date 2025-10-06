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
from commons.adk_orchestrator_agent_executor import AdkOrchestratorAgentExecutor
import logging
import os
from dotenv import load_dotenv

# Set logging
logging.getLogger().setLevel(logging.INFO)
load_dotenv()


class HostingAgentExecutor(AdkOrchestratorAgentExecutor):
    """Agent Executor that wraps OrchestratorAgentExecutor with environment-based configuration.

    This class provides backward compatibility by reading remote agent addresses
    from environment variables and delegating to OrchestratorAgentExecutor.
    """

    def __init__(self) -> None:
        """Initialize with remote agent addresses from environment variables."""
        remote_agent_addresses = [
            os.getenv("CT_AGENT_URL", "http://localhost:10002"),
            os.getenv("WEA_AGENT_URL", "http://localhost:10001"),
        ]
        super().__init__(remote_agent_addresses=remote_agent_addresses)
