from flask import Flask, request, jsonify, send_from_directory
from langchain_community.vectorstores import FAISS as FAISS_DB
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from gtts import gTTS
import speech_recognition as sr
import os
import numpy as np
import time
from pydub import AudioSegment
import tempfile
from flask_cors import CORS
from threading import Thread
import traceback
import re

import replicate
import os

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

os.makedirs("static", exist_ok=True)
# ---------------------------
# App setup
# ---------------------------
app = Flask(__name__)
CORS(app)

STATIC_FOLDER = os.path.join("mainproject", "www.stvincentngp.edu.in")

# ---------------------------
# Load models and vector store
# ---------------------------
print("🔄 Loading model and vector store...")
#llm = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", device=-1)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
db = FAISS_DB.load_local("college_vector_db", embeddings, allow_dangerous_deserialization=True)
sentiment_analyzer = pipeline("sentiment-analysis")
print("✅ Model and Vector store loaded successfully.")

# ---------------------------
# Cache setup
# ---------------------------
qa_cache = []
SIMILARITY_THRESHOLD = 0.85

def get_cached_answer(query):
    """Check if answer exists in memory cache."""
    if not qa_cache:
        return None
    query_vec = embeddings.embed_query(query)
    for item in qa_cache:
        sim = np.dot(query_vec, item["vector"]) / (
            np.linalg.norm(query_vec) * np.linalg.norm(item["vector"])
        )
        if sim >= SIMILARITY_THRESHOLD:
            return item["answer"]
    return None

def store_in_cache(question, answer):
    """Store answer and its vector in cache."""
    q_vec = embeddings.embed_query(question)
    qa_cache.append({"question": question, "answer": answer, "vector": q_vec})

# ---------------------------
# Serve static files (frontend)
# ---------------------------
@app.route("/")
def serve_index():
    return send_from_directory(STATIC_FOLDER, "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(STATIC_FOLDER, path)

# ---------------------------
# Voice Query Route
# ---------------------------
@app.route("/voice", methods=["POST"])
def voice_chat():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio uploaded"}), 400

        audio_file = request.files["audio"]

        # Save uploaded webm to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_input:
            audio_file.save(temp_input.name)
            temp_input_path = temp_input.name

        # Convert webm -> wav
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            temp_wav_path = temp_wav.name

        sound = AudioSegment.from_file(temp_input_path)
        sound.export(temp_wav_path, format="wav")

        # Recognize speech
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_wav_path) as source:
            audio_data = recognizer.record(source)

        query = recognizer.recognize_google(audio_data)
        print("🎤 User said:", query)

        # Generate answer
        answer = generate_answer(query)

        # Clean up temp files
        try:
            os.remove(temp_input_path)
            os.remove(temp_wav_path)
        except Exception as cleanup_e:
            print("⚠️ Could not remove temp files:", cleanup_e)

        return jsonify(answer)

    except Exception as e:
        print("❌ Error in /voice:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------------------------
# Core Answer Function
# ---------------------------
def _is_clarifying_response(text: str) -> bool:
    """Heuristic to detect if LLM returned a clarifying question."""
    if not text:
        return True
    t = text.strip()
    starts_with_q = bool(re.match(r'^(what|which|who|when|where|why|how|please|could|can|would)\b', t.lower()))
    short_len = len(t) < 140
    question_count = t.count('?')
    first_sentence = t.split('\n')[0]
    first_sentence_has_q = '?' in first_sentence and len(first_sentence) < 120

    if (starts_with_q and short_len) or (question_count > 0 and first_sentence_has_q):
        return True
    if re.match(r'^(i need|please provide|could you|please tell|i don\'t have)', t.lower()):
        return True
    return False
    
def call_mistral_replicate(prompt):
    try:
        output = replicate.run(
            "mistralai/mistral-7b-instruct-v0.1",
            input={
                "prompt": prompt,
                "temperature": 0.1,
                "max_new_tokens": 200
            }
        )
        return "".join(output)
    except Exception as e:
        print("❌ Replicate API error:", e)
        return "Sorry, the model failed to generate a response."

def generate_answer(query):
    """Generate an answer (used by both voice and text routes)."""
    try:
        sentiment_result = sentiment_analyzer(query)[0]
        sentiment_label = sentiment_result.get('label', '')
        greeting = {
            "POSITIVE": "I’m glad to hear from you! ",
        }.get(sentiment_label, "Okay, let's see what I can find for you. ")

        cached = get_cached_answer(query)
        if cached:
            final_answer = greeting + f"(From memory) {cached}"
            audio_url = None
            try:
                os.makedirs("static", exist_ok=True)
                audio_filename = f"response_{int(time.time())}.mp3"
                tts = gTTS(final_answer)
                tts.save(os.path.join("static", audio_filename))
                audio_url = f"/static/{audio_filename}"
            except Exception as tts_e:
                print("⚠️ gTTS failed for cached answer:", tts_e)
            return {"query": query, "answer": final_answer, "audio_url": audio_url}

        docs = db.similarity_search(query, k=8)
        print("🔎 Retrieved docs count:", len(docs))
        for i, d in enumerate(docs[:4]):
            preview = d.page_content.replace("\n", " ")[:300]
            print(f"  doc[{i}] preview: {preview!r}...")

        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"""
You are Navi, a helpful assistant for a college website.
ONLY answer based on the context below.
If the answer is not present in the context, reply exactly:
"Sorry, I could not find that information."

Context:
{context}

User: {query}
Answer:
"""

        print("🧠 Calling Mistral-7B on Replicate…")
        answer_text = call_mistral_replicate(prompt).strip()
        print("📝 Raw model output preview:", answer_text[:300])

        if _is_clarifying_response(answer_text):
            print("⚠️ Model returned a clarifying question — converting to 'not found' fallback.")
            answer_text = "Sorry, I could not find that information."

        if "Sorry, I could not find that information." not in answer_text:
            store_in_cache(query, answer_text)

        final_answer = greeting + answer_text

         # TTS
        os.makedirs("static", exist_ok=True)
        audio_filename = f"tts_{int(time.time())}.mp3"
        audio_path = os.path.join("static", audio_filename)
        try:
            tts = gTTS(final_answer)
            tts.save(audio_path)
            audio_url = f"/static/{audio_filename}"
        except Exception as tts_e:
            print("⚠️ TTS failed:", tts_e)
            audio_url = None

        return {"query": query, "answer": final_answer, "audio_url": audio_url}


    except Exception as e:
        print("Error in generate_answer():", e)
        traceback.print_exc()
        return {"error": str(e)}

# ---------------------------
# Background job system (Text Queries)
# ---------------------------
tasks = {}

def process_query(task_id, query):
    """Run heavy model in background safely."""
    try:
        response = generate_answer(query)
        if "error" in response:
            tasks[task_id] = {"status": "error", "error": response["error"]}
        else:
            tasks[task_id] = {
                "status": "done",
                "answer": response["answer"],
                "audio_url": response["audio_url"],
            }
        print(f"✅ Finished processing task {task_id}")
    except Exception as e:
        print("Error in background process_query():", e)
        traceback.print_exc()
        tasks[task_id] = {"status": "error", "error": str(e)}

@app.route("/ask", methods=["POST"])
def ask():
    """Starts background LLM query job."""
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    task_id = str(int(time.time()))
    tasks[task_id] = {"status": "processing"}
    Thread(target=process_query, args=(task_id, query), daemon=True).start()
    print(f"🚀 Started background task {task_id} for query: {query}")
    return jsonify({"task_id": task_id})

@app.route("/result/<task_id>")
def get_result(task_id):
    """Fetch background job result."""
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "Invalid task ID"}), 404
    return jsonify(task)

# ---------------------------
# Run App
# ---------------------------
if __name__ == "__main__":
    print("🚀 Flask chatbot running at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)

