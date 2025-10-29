import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from pathlib import Path

# =======================
# Load Data
# =======================
base_dir = Path(__file__).resolve().parent
csv_path = base_dir / "health_data.csv"

if not csv_path.exists():
    csv_path = Path.cwd() / "health_data.csv"

if not csv_path.exists():
    st.error("⚠️ Could not find 'health_data.csv'. Make sure it's in the same folder.")
    st.stop()

data = pd.read_csv(csv_path)
data.dropna(inplace=True)

# =======================
# Load Sentence Transformer
# =======================
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-MiniLM-L6-v2")

model = load_model()

# Precompute embeddings for all symptoms in CSV
symptom_embeddings = model.encode(data["Symptoms"].tolist(), convert_to_tensor=True)

# =======================
# Streamlit Page Setup
# =======================
st.set_page_config(page_title="AI Health Assistant 🧠", page_icon="💊", layout="centered")

st.markdown("""
    <h1 style='text-align:center; color:#1e40af;'>🧠 AI Health Assistant</h1>
    <p style='text-align:center; color:#3b82f6; font-size:18px;'>
        Get smart health advice powered by AI similarity search
    </p>
""", unsafe_allow_html=True)

# =======================
# User Input
# =======================
user_input = st.text_area("Enter your symptoms (e.g., headache, fever, nausea):", height=100)

# =======================
# AI Advice Logic
# =======================
if st.button("💡 Get Health Advice"):
    if not user_input.strip():
        st.warning("⚠️ Please enter your symptoms first.")
    else:
        # Split input into multiple symptoms
        user_symptoms = [s.strip() for s in user_input.split(",") if s.strip()]

        # Encode all entered symptoms
        user_embeddings = model.encode(user_symptoms, convert_to_tensor=True)

        # Find most similar symptom from dataset
        scores = util.cos_sim(user_embeddings, symptom_embeddings)

        # Get top match for each entered symptom
        results = []
        for i, symptom in enumerate(user_symptoms):
            best_match_idx = scores[i].argmax().item()
            best_score = scores[i][best_match_idx].item()
            matched_symptom = data.iloc[best_match_idx]["Symptoms"]
            advice = data.iloc[best_match_idx]["Advice"]

            if best_score > 0.45:  # similarity threshold
                results.append(f"🩺 **{symptom.title()} → {matched_symptom}:** {advice}")
            else:
                results.append(f"❗ No close match found for '{symptom}'. Please consult a doctor.")

        # Show results
        st.markdown("<div style='background-color:#e0f2fe; padding:15px; border-radius:10px;'>"
                    + "<br>".join(results) + "</div>", unsafe_allow_html=True)

# =======================
# Sidebar
# =======================
st.sidebar.title("About Project")
st.sidebar.info("""
### 🧬 AI Health Assistant
- Built using **Python**, **Streamlit**, and **Sentence Transformers**
- Gives smart medical suggestions for multiple symptoms
- Developed by **Team INSANE** | Velammal Vidyalaya Avadi
""")
