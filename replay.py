import asyncio
import json
from main import telegram_webhook

# Minimal fake Request to pass to the handler
class FakeRequest:
    def __init__(self, payload: dict):
        self._payload = payload
    async def json(self):
        return self._payload

async def main():
    # Load a recorded Telegram update (ndjson line or a JSON file)
    with open("update_yes_trip.json", "r", encoding="utf-8") as f:
        payload = json.load(f)  # first logged update

    req = FakeRequest(payload)
    result = await telegram_webhook(req)  # set breakpoints inside telegram_webhook/graph
    print(result)

if __name__ == "__main__":
    asyncio.run(main())