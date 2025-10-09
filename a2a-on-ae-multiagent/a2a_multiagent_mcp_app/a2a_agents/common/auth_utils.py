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
"""Shared authentication utilities for Google Cloud services."""

import logging
from typing import Generator

import httpx
from google.auth import default
from google.auth.credentials import Credentials
from google.auth.transport.requests import Request as AuthRequest

logger = logging.getLogger(__name__)


class GoogleAuth(httpx.Auth):
    """A custom httpx Auth class for Google Cloud authentication.

    This class implements httpx's Auth interface to automatically handle
    Google Cloud authentication by:
    1. Using Application Default Credentials (ADC)
    2. Automatically refreshing expired tokens
    3. Adding the Authorization header to all requests

    Example:
        >>> client = httpx.AsyncClient(auth=GoogleAuth())
        >>> response = await client.get("https://example.googleapis.com/api")
    """

    def __init__(self) -> None:
        """Initializes the GoogleAuth instance with default credentials.

        Uses Application Default Credentials with cloud-platform scope.
        This will automatically use:
        - Service account credentials in production
        - User credentials from gcloud auth in development
        - Metadata server credentials in GCP environments
        """
        self.credentials: Credentials
        self.project: str | None
        self.credentials, self.project = default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self.auth_request = AuthRequest()

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, None, None]:
        """Adds the Authorization header to the request.

        This method is called by httpx for each request. It ensures that:
        1. The credentials are valid and refreshes them if expired
        2. The Authorization header is added with the current token

        Args:
            request: The httpx request to add the header to.

        Yields:
            The request with the Authorization header added.
        """
        # Refresh the credentials if they are expired
        if not self.credentials.valid:
            logger.info("Credentials expired, refreshing...")
            self.credentials.refresh(self.auth_request)

        # Add the Authorization header to the request
        request.headers["Authorization"] = f"Bearer {self.credentials.token}"
        yield request
