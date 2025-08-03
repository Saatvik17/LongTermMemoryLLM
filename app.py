from flask import Flask, request, jsonify
from agent import MemoryAgent

app = Flask(__name__)
agent = MemoryAgent()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    if not data or "message" not in data:
        return jsonify({"error": "Error in chat request."}), 400
    response = agent.process(data["message"])
    return jsonify({"response": response})

@app.route("/memories", methods=["GET"])
def list_memories():
    return jsonify(agent.store.list_all())

@app.route("/memory", methods=["DELETE"])
def delete_memory():
    data = request.json
    if not data or "keyword" not in data:
        return jsonify({"error": "Error in memory request."}), 400
    count = agent.store.delete(data["keyword"])
    return jsonify({"deleted": count})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
