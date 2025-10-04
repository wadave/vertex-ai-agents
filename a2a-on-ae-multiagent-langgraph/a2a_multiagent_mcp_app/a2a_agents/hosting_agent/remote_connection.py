"""Remote connections helper."""

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

import traceback

from collections.abc import Callable

from a2a.client import (
    Client,
    ClientFactory,
)
from a2a.types import (
    AgentCard,
    Message,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
)


TaskCallbackArg = Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
TaskUpdateCallback = Callable[[TaskCallbackArg, AgentCard], Task]


class RemoteAgentConnections:
    """A class to hold the connections to the remote agents."""

    def __init__(self, client_factory: ClientFactory, agent_card: AgentCard):
        self.agent_client: Client = client_factory.create(agent_card)
        self.card: AgentCard = agent_card
        self.pending_tasks = set()

    def get_agent(self) -> AgentCard:
        return self.card

    async def send_message(self, message: Message) -> Task | Message | None:
        lastTask: Task | None = None
        try:
            print('Sending message to remote agent:', message)
            async for event in self.agent_client.send_message(message):
                if isinstance(event, Message):
                    print('got event object from remote agent:', event)
                    return event
                if self.is_terminal_or_interrupted(event[0]):
                    return event[0]
                lastTask = event[0]
        except Exception as e:
            print('Exception found in send_message')
            traceback.print_exc()
            raise e
        return lastTask

    def is_terminal_or_interrupted(self, task: Task) -> bool:
        return task.status.state in [
            TaskState.completed,
            TaskState.canceled,
            TaskState.failed,
            TaskState.input_required,
            TaskState.unknown,
        ]