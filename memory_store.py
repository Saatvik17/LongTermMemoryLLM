import sqlite3, pickle, numpy as np

class MemoryStore:
    def __init__(self, db_path="memories.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT,
            content TEXT,
            embedding BLOB
        )""")
        self.conn.commit()

    def add(self, category, content, embedding):
        self.conn.execute(
            "INSERT INTO memories (category, content, embedding) VALUES (?, ?, ?)",
            (category, content, pickle.dumps(embedding))
        )
        self.conn.commit()

    def delete(self, keyword):
        cur = self.conn.execute("DELETE FROM memories WHERE content LIKE ?", (f"%{keyword}%",))
        count = cur.rowcount
        self.conn.commit()
        return count

    def list_all(self):
        cur = self.conn.execute("SELECT id, category, content FROM memories")
        return [{"id": r[0], "category": r[1], "content": r[2]} for r in cur.fetchall()]

    def search(self, query_emb, top_k=3):
        cur = self.conn.execute("SELECT id, category, content, embedding FROM memories")
        results = []
        for row in cur.fetchall():
            emb = pickle.loads(row[3])
            sim = float(np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb)))
            results.append((row[1], row[2], sim))
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]