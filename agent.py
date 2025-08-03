from embedder import Embedder
from memory_store import MemoryStore
from llm_client import LLMClient
from intent_detector import IntentDetector

class MemoryAgent:
    def __init__(self):
        self.embedder = Embedder()
        self.store = MemoryStore()
        self.llm = LLMClient()
        self.detector = IntentDetector()

    def process(self, user_msg: str) -> str:
        intent_data = self.detector.detect(user_msg)
        intent = intent_data.get("intent")

        if intent == "add":
            category = intent_data.get("category", "general")
            content = intent_data.get("content", user_msg)
            emb = self.embedder.embed(content)
            self.store.add(category, content, emb)
            return f"Added memory: {content}"

        elif intent == "delete":
            keyword = intent_data.get("content", "")
            count = self.store.delete(keyword)
            return f"Deleted {count} memory item(s)." if count else "No matching memory."

        elif intent == "query":
            query = intent_data.get("query", user_msg)
            results = self.store.search(self.embedder.embed(query))
            if not results:
                return "I have no memory about that."
            memories = ", ".join(f"{r[1]}" for r in results)
            return f"Based on memory: {memories}"

        # Contextual response
        relevant = self.store.search(self.embedder.embed(user_msg))
        context = "Relevant memories:\n" + "\n".join(f"- {m[1]}" for m in relevant)
        return self.llm.generate(f"{context}\nUser: {user_msg}\nAssistant:")
