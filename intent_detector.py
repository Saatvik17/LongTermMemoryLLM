import json
from llm_client import LLMClient

class IntentDetector:
    def __init__(self):
        self.llm = LLMClient()

    def detect(self, message: str):
        prompt = f"""
You are an AI that extracts memory operations.

Message: "{message}"

Decide what the user wants:
- If they are **adding** memory, return JSON: {{"intent": "add", "category": "...", "content": "..."}}
- If they are **deleting**, return JSON: {{"intent": "delete", "content": "..."}}
- If they are **asking a question**, return JSON: {{"intent": "query", "query": "..."}}
- Otherwise, return JSON: {{"intent": "chat"}}

Respond **only with JSON**, no extra text.
"""
        result = self.llm.generate(prompt)
        try:
            return json.loads(result.strip().split("\n")[-1])
        except:
            return {"intent": "chat"}
