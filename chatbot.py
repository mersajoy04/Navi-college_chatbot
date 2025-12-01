from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import numpy as np
import speech_recognition as sr
import tempfile
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ---------------- Model + Vector Store ----------------
print("🔄 Loading Mistral model...")
#llm = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", device=-1)
llm = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.1",
    use_auth_token="YOUR_HF_TOKEN"
)

print("✅ Model loaded.")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
db = FAISS.load_local("college_vector_db", embedding_model, allow_dangerous_deserialization=True)
print("✅ Vector store loaded.")

qa_cache = []
SIMILARITY_THRESHOLD = 0.85


def get_cached_answer(query):
    if not qa_cache:
        return None
    query_vec = embedding_model.embed_query(query)
    for item in qa_cache:
        sim = np.dot(query_vec, item["vector"]) / (np.linalg.norm(query_vec) * np.linalg.norm(item["vector"]))
        if sim >= SIMILARITY_THRESHOLD:
            return item["answer"]
    return None


def store_in_cache(q, a):
    qa_cache.append({"question": q, "answer": a, "vector": embedding_model.embed_query(q)})


def get_answer(query):
    query = query.strip()
    cached = get_cached_answer(query)
    if cached:
        return f"(From memory) {cached}"

    docs = db.similarity_search(query, k=8)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a helpful assistant for a college website.
ONLY answer based on the context below.
If not found, say "Sorry, I could not find that information."

Context:
{context}

User: {query}
Answer:
"""
    result = llm(prompt, max_new_tokens=150, do_sample=True)
    answer = result[0]["generated_text"].split("Answer:")[-1].strip()
    store_in_cache(query, answer)
    return answer


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        query = data.get("query", "")
        if not query.strip():
            return jsonify({"error": "Empty query"}), 400
        answer = get_answer(query)
        return jsonify({"answer": answer})
    except Exception as e:
        print("Error in /ask:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/voice", methods=["POST"])
def voice():
    """Handles audio input from the mic."""
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file received"}), 400

        file = request.files["audio"]
        temp_path = os.path.join(tempfile.gettempdir(), "voice_input.wav")
        file.save(temp_path)

        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_path) as source:
            audio = recognizer.record(source)
            query = recognizer.recognize_google(audio)
        os.remove(temp_path)

        answer = get_answer(query)
        return jsonify({"query": query, "answer": answer})
    except Exception as e:
        print(" Error in /voice:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Chatbot running successfully!"})


if __name__ == "__main__":
    print("🚀 Flask chatbot running at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
