# app.py
import streamlit as st
from utils import load_pdfs_from_folder_with_page_metadata, chunk_documents_with_metadata, create_faiss_index, get_embedder
from groq import Groq
import random
import re

# Setup Groq API
GROQ_API_KEY = "XXXXXXXXXXXXX" # Replace with your actual Groq API key
MODEL_NAME = "YourModelName"  # Replace with your actual model name
client = Groq(api_key=GROQ_API_KEY)

st.title("🧠 PDF Quiz Generator with MCQs")

# Load and process documents
if "vectorstore" not in st.session_state:
    with st.spinner("Loading and processing documents..."):
        raw_docs = load_pdfs_from_folder_with_page_metadata("pdfs")
        docs = chunk_documents_with_metadata(raw_docs)
        embedder = get_embedder()
        db = create_faiss_index(docs, embedder)
        st.session_state.vectorstore = db
        st.session_state.documents = docs
        st.session_state.embedder = embedder

# Generate quiz questions
if st.button("Generate 10 MCQ Questions"):
    with st.spinner("Generating questions..."):
        questions = []
        docs = st.session_state.documents
        sampled_docs = random.sample(docs, min(10, len(docs)))

        for i, doc in enumerate(sampled_docs):
            prompt = f"""
You are a helpful assistant creating a **multiple-choice question** (MCQ) for a quiz from the following context.
Generate one question with 4 options (A, B, C, D).
Clearly indicate which option is correct.
Also, provide the page number from which this question is derived.

Context:
{doc.page_content}

Format your response exactly as:
Question: <your question here>
Options:
A) option 1
B) option 2
C) option 3
D) option 4
Answer: <Correct option letter>
Reference: Page {doc.metadata.get('page', 'unknown')}

---
"""

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                )
                quiz_text = response.choices[0].message.content

                q_match = re.search(r"Question:\s*(.*)", quiz_text)
                a_matches = re.findall(r"([ABCD])\)\s*(.*)", quiz_text)
                answer_match = re.search(r"Answer:\s*([ABCD])", quiz_text)
                ref_match = re.search(r"Reference:\s*Page\s*(\d+)", quiz_text)

                if not (q_match and len(a_matches) == 4 and answer_match):
                    raise ValueError("Parsing failed: missing fields")

                q_text = q_match.group(1).strip()
                options = {letter: text.strip() for letter, text in a_matches}
                answer = answer_match.group(1).strip()
                reference = f"Page {ref_match.group(1)}" if ref_match else f"Page {doc.metadata.get('page', 'unknown')}"

                questions.append({
                    "question": q_text,
                    "options": options,
                    "answer": answer,
                    "reference": reference,
                })

            except Exception as e:
                st.warning(f"Failed to parse question #{i+1}: {e}")

        st.session_state.quiz_questions = questions
        st.session_state.user_answers = {}

# Quiz UI
if "quiz_questions" in st.session_state:
    st.header("📝 Answer the Quiz")
    for i, q in enumerate(st.session_state.quiz_questions):
        st.write(f"**Q{i+1}. {q['question']}**")
        user_choice = st.radio(
            f"Select an option for Q{i+1}",
            options=["A", "B", "C", "D"],
            format_func=lambda x: f"{x}) {q['options'][x]}",
            key=f"q{i}"
        )
        st.session_state.user_answers[f"q{i}"] = user_choice

    if st.button("Submit Answers"):
        score = 0
        results = []
        for i, q in enumerate(st.session_state.quiz_questions):
            user_ans = st.session_state.user_answers.get(f"q{i}")
            correct = q['answer']
            is_correct = (user_ans == correct)
            if is_correct:
                score += 1
            results.append({
                "question": q['question'],
                "your_answer": user_ans,
                "correct_answer": correct,
                "correct_option_text": q['options'][correct],
                "reference": q['reference'],
                "is_correct": is_correct
            })

        st.subheader(f"✅ Your Score: {score} / {len(st.session_state.quiz_questions)}")

        for i, r in enumerate(results):
            st.markdown(f"**Q{i+1}: {r['question']}**")
            st.markdown(f"- Your answer: {r['your_answer']}) {q['options'][r['your_answer']]}" if r['your_answer'] in q['options'] else "- Your answer: Not selected")
            st.markdown(f"- Correct answer: {r['correct_answer']}) {r['correct_option_text']}")
            st.markdown(f"- Reference: {r['reference']}")
            if r["is_correct"]:
                st.success("✅ Correct")
            else:
                st.error("❌ Incorrect")
