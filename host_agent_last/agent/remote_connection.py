from typing import Optional
import httpx
from a2a.client import A2AClient
from a2a.types import AgentCard, SendMessageRequest, SendMessageResponse

class RemoteAgentConnections:
    """No HTTP timeouts (HTTPX). Still uses A2AClient."""
    def __init__(self, agent_card: AgentCard, agent_url: str):
        print(f"agent_card: {agent_card}")
        print(f"agent_url: {agent_url}")

        # Disable all HTTPX timeouts: connect/read/write/pool
        timeout = httpx.Timeout(None)  # <- NO TIMEOUTS
        limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
        self._httpx_client = httpx.AsyncClient(timeout=timeout, limits=limits, headers={"Connection": "keep-alive"})

        self.agent_client = A2AClient(self._httpx_client, agent_card, url=agent_url)
        self.card = agent_card
        self.conversation_name: Optional[str] = None
        self.conversation = None
        self.pending_tasks = set()

    def get_agent(self) -> AgentCard:
        return self.card

    async def send_message(self, message_request: SendMessageRequest) -> SendMessageResponse:
        # NOTE: If your A2AClient version applies its *own* timeout internally,
        # switch to Option B below (direct HTTP).
        return await self.agent_client.send_message(message_request)
