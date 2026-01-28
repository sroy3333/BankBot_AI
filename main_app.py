# main_app.py
import re
import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import json
import sys
import subprocess
import uuid
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from database.bank_crud import create_account, delete_account

from nlu_engine.infer_intent import IntentClassifier
from nlu_engine.entity_extractor import EntityExtractor
from dialogue_manager.stories import INTENT_STORIES
from dialogue_manager.dialogue_handler import DialogueManager
from dialogue_manager.dialogue_state import DialogueState
from router.query_router import QueryRouter
from services.llm_service import ask_llm
from admin_panel import admin_panel
from services.chat_logger import log_chat
from services.safety_guard import looks_like_banking
from database.db import get_conn
from datetime import datetime

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))

# ---------------------------------------------------
# 🔐 GLOBAL AUTH STATE
# ---------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

if "username" not in st.session_state:
    st.session_state.username = None


# ---------------------------------------------------
# Paths
# ---------------------------------------------------
INTENTS_PATH = "nlu_engine/intents.json" 
MODEL_DIR = "models/intent_model"
LOG_PATH = os.path.join("models", "training.log")

os.makedirs("models", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------
# Page Config + Theme
# ---------------------------------------------------
st.set_page_config(
    page_title="BankBot NLU",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>

/* ---------- Global App ---------- */
.stApp {
    background: radial-gradient(circle at top left, #0f172a, #020617);
    color: #e5e7eb;
}

/* ---------- Titles ---------- */
.app-title {
    font-size: 42px;
    font-weight: 800;
    color: #38bdf8;
}

.app-subtitle {
    font-size: 16px;
    color: #94a3b8;
    margin-bottom: 25px;
}

/* ---------- Cards ---------- */
.card {
    background: linear-gradient(145deg, #020617, #020617);
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 20px;
    box-shadow:
        0 0 0 1px rgba(56,189,248,0.15),
        0 20px 40px rgba(0,0,0,0.6);
}

/* ---------- Section Titles ---------- */
.section-title {
    font-size: 20px;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 14px;
}

/* ---------- Text Inputs ---------- */
textarea, input {
    background-color: #020617 !important;
    color: #e5e7eb !important;
    border-radius: 10px !important;
    border: 1px solid #1e293b !important;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: linear-gradient(135deg, #38bdf8, #0ea5e9);
    color: #020617;
    font-weight: 700;
    border-radius: 12px;
    padding: 0.6rem 1.2rem;
    border: none;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #0ea5e9, #0284c7);
}

/* ---------- Metrics ---------- */
[data-testid="stMetricValue"] {
    color: #38bdf8;
    font-size: 28px;
    font-weight: 800;
}

/* ---------- Success Entity Box ---------- */
.success-box {
    background: linear-gradient(135deg, #022c22, #064e3b);
    border: 1px solid #10b981;
    padding: 14px;
    border-radius: 12px;
    color: #a7f3d0;
    font-weight: 600;
    margin-bottom: 10px;
}

/* ---------- Tables ---------- */
.stDataFrame, .stTable {
    background-color: #020617 !important;
    border-radius: 14px;
    border: 1px solid #1e293b;
}

/* ---------- Expander ---------- */
details {
    background-color: #020617 !important;
    border-radius: 14px !important;
    border: 1px solid #1e293b !important;
}

/* ---------- Progress Bars ---------- */
div[role="progressbar"] > div {
    background: linear-gradient(90deg, #38bdf8, #0ea5e9);
}

/* ---------- Scrollbar ---------- */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: #1e293b;
    border-radius: 6px;
}

/* ---------- Footer ---------- */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# 🔐 GLOBAL LOGIN GATE
# ---------------------------------------------------
if not st.session_state.authenticated:

    st.markdown("## 🔐 BankBot Secure Login")
    st.caption("Username and password required")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

    if submit:
        if (
            username == os.getenv("ADMIN_USERNAME")
            and password == os.getenv("ADMIN_PASSWORD")
        ):
            st.session_state.authenticated = True
            st.session_state.is_admin = True
            st.session_state.username = username

            st.success(f"✅ Welcome, {username}")
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

    # ⛔ BLOCK ENTIRE APP
    st.stop()


# ---------------------------------------------------
# Header
# ---------------------------------------------------
st.markdown("🏦 <span class='app-title'>BankBot NLU Engine</span>", unsafe_allow_html=True)
st.markdown(
    "<div class='app-subtitle'>Secure Banking Intent Intelligence Platform</div>",
    unsafe_allow_html=True
)

# ---------------------------------------------------
# Side-bar Login State
# ---------------------------------------------------
with st.sidebar:
    if st.session_state.authenticated:
        st.markdown("### 👤 Session")

        # ⏰ Time-based greeting
        hour = datetime.now().hour
        if hour < 12:
            greet = "Good morning"
        elif hour < 18:
            greet = "Good afternoon"
        else:
            greet = "Good evening"

        display_name = st.session_state.username.capitalize()
        st.success(f"👋 {greet} **{display_name}**")

        # 🚪 Logout
        if st.button("🚪 Logout"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()



# ---------------------------------------------------
# Utilities
# ---------------------------------------------------
def load_intents_file():
    if not os.path.exists(INTENTS_PATH):
        return []
    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data if isinstance(data, list) else data.get("intents", [])
    

def sync_training_data(intents):
    conn = get_conn()
    cur = conn.cursor()

    # Clear existing data
    cur.execute("DELETE FROM training_data")

    # Insert latest intent examples
    for intent in intents:
        tag = intent.get("tag")
        for example in intent.get("examples", []):
            cur.execute(
                "INSERT INTO training_data (intent, example) VALUES (?, ?)",
                (tag, example)
            )

    conn.commit()
    conn.close()


def save_intents_file(intents):
    with open(INTENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(intents, f, indent=2, ensure_ascii=False)

    # 🔥 Sync intents.json → SQLite for Admin Panel
    sync_training_data(intents)

# ---------------------------------------------------
# DB init + startup sync
# ---------------------------------------------------

from database.db import init_db
init_db()

intents = load_intents_file()
if intents:
    sync_training_data(intents)


def start_training_subprocess(epochs, batch_size, lr):
    cmd = [
        sys.executable, "nlu_engine/train_intent.py",
        "--data_path", INTENTS_PATH,
        "--model_dir", MODEL_DIR,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(lr),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# ---------------------------------------------------
# Tabs
# ---------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["🧠 NLU Analyzer", "🛠️ Intent Manager", "📜 History", "📊 Model Evaluation", "🤖 Bank Assistant", "🏗️ Account Management", "🔐 Admin Panel"]
)

# ===================================================
# 🧠 TAB 1 — NLU ANALYZER
# ===================================================
from database.logger import log_user_query

def handle_analyze_query():
    query = st.session_state.nlu_query.strip()
    if not query:
        st.warning("Please enter a query")
        return

    classifier = IntentClassifier(MODEL_DIR)
    extractor = EntityExtractor()

    predictions = classifier.predict(query, st.session_state.top_k)
    top_intent = predictions[0]

    # ---------- HISTORY----------
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "query": query,
        "intent": top_intent["label"],
        "confidence": top_intent["score"]
    })

    #----------------- LOGGING ------------------------
    log_user_query(
        user_input=query,
        intent=top_intent["label"],
        confidence=top_intent["score"]
    )

    log_chat(
        session_id=st.session_state.session_id,
        role="user",
        message=f"[NLU_ANALYZER] {query}"
    )

    # ---------- STORE RESULT ----------
    st.session_state.nlu_result = {
        "query": query,
        "predictions": predictions,
        "top_intent": top_intent,
        "entities": extractor.extract(query)
    }

    st.session_state.nlu_query = ""

    st.rerun()

with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Analyze Customer Query</div>", unsafe_allow_html=True)

    st.text_area(
        "Enter query",
        placeholder="Transfer ₹5000 from savings account to account 123456",
        height=90,
        key="nlu_query"
    )

    st.slider(
        "Top K Intents",
        1, 10, 4,
        key="top_k"
    )

    st.button(
        "Analyze Query",
        on_click=handle_analyze_query
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if "nlu_result" in st.session_state:
        result = st.session_state.nlu_result
        predictions = result["predictions"]
        top_intent = result["top_intent"]
        entities = result["entities"]

        # ---------- TOP INTENT ----------
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"**Top Intent:** `{top_intent['label']}`")
        st.progress(int(top_intent["score"] * 100))
        st.markdown(f"Confidence: **{top_intent['score']*100:.2f}%**")
        st.caption("⚠ Confidence ≠ Accuracy. Accuracy is shown in the Model Evaluation tab.")
        st.markdown("</div>", unsafe_allow_html=True)

        # ---------- INTENT RANKING ----------
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Intent Ranking</div>", unsafe_allow_html=True)

        scores = [p["score"] for p in predictions]
        min_s, max_s = min(scores), max(scores)

        def normalize(score, min_s, max_s):
            if max_s == min_s:
                return 0.15
            return 0.1 + (score - min_s) / (max_s - min_s) * 0.1

        for p in predictions:
            norm_score = normalize(p["score"], min_s, max_s)
            st.markdown(f"**{p['label']} — {norm_score:.3f}**")
            st.progress(int(norm_score * 100))

        st.markdown("</div>", unsafe_allow_html=True)

        # ---------- ENTITY EXTRACTION ----------
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Extracted Entities</div>", unsafe_allow_html=True)

        if entities:
            for e in entities:
                st.markdown(
                    f"<div class='success-box'>{e['entity']} : {e['value']}</div>",
                    unsafe_allow_html=True
                )
        else:
            st.info("No entities found")

        st.markdown("</div>", unsafe_allow_html=True)

# ===================================================
# 🛠️ TAB 2 — INTENT MANAGER
# ===================================================
with tab2:
    # ==============================
    # Manage Intents
    # ==============================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("🛠️ <span class='section-title'>Manage Intents</span>", unsafe_allow_html=True)

    intents = load_intents_file()
    updated_intents = []

    for i, intent in enumerate(intents):
        with st.expander(f"{intent['tag']} ({len(intent['examples'])} examples)"):
            examples_text = st.text_area(
                label="Examples (one per line)",
                value="\n".join(intent["examples"]),
                height=130,
                key=f"edit_intent_{i}"  # ✅ unique key
            )

            updated_intents.append({
                "tag": intent["tag"],
                "examples": [e.strip() for e in examples_text.splitlines() if e.strip()]
            })

    if st.button("Save Intent Changes"):
        save_intents_file(updated_intents)
        st.success("Intents updated successfully ✅")

    st.markdown("</div>", unsafe_allow_html=True)

    # ==============================
    # Add New Intent
    # ==============================

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("➕ <span class='section-title'>Add New Intent</span>", unsafe_allow_html=True)

    # Initialize state safely
    if "new_intent_name" not in st.session_state:
        st.session_state.new_intent_name = ""
    if "new_intent_examples" not in st.session_state:
        st.session_state.new_intent_examples = ""

    new_intent_name = st.text_input(
        "Intent Name",
        key="new_intent_name"
    )

    new_intent_examples = st.text_area(
        "Examples (one per line)",
        height=140,
        key="new_intent_examples"
    )

    def save_new_intent():
        intent_name = st.session_state.new_intent_name.strip()
        example_lines = [
            e.strip()
            for e in st.session_state.new_intent_examples.splitlines()
            if e.strip()
        ]

        if not intent_name or not example_lines:
            st.error("Intent name and at least one example are required")
            return

        if not re.match(r"^[a-z_]+$", intent_name):
            st.error("Intent name must be lowercase snake_case (e.g. transfer_money)")
            return

        intents = load_intents_file()

        existing = next((i for i in intents if i["tag"] == intent_name), None)

        if existing:
            for ex in example_lines:
                if ex not in existing["examples"]:
                    existing["examples"].append(ex)
        else:
            intents.append({
                "tag": intent_name,
                "examples": example_lines
            })

        save_intents_file(intents)

        # ✅ SAFE reset (inside callback)
        st.session_state.new_intent_name = ""
        st.session_state.new_intent_examples = ""

        st.success(f"Intent '{intent_name}' saved successfully ✅")

    st.button("Save Intent", on_click=save_new_intent)

    st.markdown("</div>", unsafe_allow_html=True)


    # ---------- TRAINING ----------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Train Model</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    epochs = c1.number_input("Epochs", 1, 50, 5)
    batch_size = c2.number_input("Batch Size", 1, 128, 4)
    lr = c3.number_input("Learning Rate", 1e-7, 1.0, 2e-5, format="%.6f")

    if st.button("Start Training"):
        with st.spinner("Training model..."):
            proc = start_training_subprocess(epochs, batch_size, lr)
            for line in proc.stdout:
                st.text(line.strip())
            proc.wait()

            if proc.returncode == 0:
                st.success("Training completed successfully")
            else:
                st.error("Training failed")
                st.text(proc.stderr.read())

    st.markdown("</div>", unsafe_allow_html=True)

# ===================================================
# 📜 TAB 3 — HISTORY
# ===================================================
with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Query History</div>", unsafe_allow_html=True)

    history = st.session_state.get("history", [])

    if st.button("Clear History"):
        st.session_state.history = []
        history = []

    if not history:
        st.info("No queries analyzed yet")
    else:
        for h in reversed(history):
            st.markdown(f"""
            **Query:** {h['query']}  
            Intent: `{h['intent']}`  
            Confidence: {h['confidence']:.2f}
            ---
            """)

    st.markdown("</div>", unsafe_allow_html=True)


# ===================================================
# 📊 TAB 4 — MODEL EVALUATION
# ===================================================
with tab4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Model Evaluation</div>", unsafe_allow_html=True)

    if not os.path.exists(os.path.join(MODEL_DIR, "model.pkl")):
        st.warning("Model not trained yet.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    # Load intents
    intents = load_intents_file()

    texts, true_labels = [], []
    for intent in intents:
        for ex in intent["examples"]:
            texts.append(ex)
            true_labels.append(intent["tag"])

    classifier = IntentClassifier(MODEL_DIR)

    # Predict labels
    preds = []
    for t in texts:
        pred = classifier.predict(t, top_k=1)[0]["label"]
        preds.append(pred)

    # Accuracy
    acc = accuracy_score(true_labels, preds)

    st.metric("Model Accuracy", f"{acc*100:.2f}%")

    # Confusion Matrix
    labels = sorted(list(set(true_labels)))
    cm = confusion_matrix(true_labels, preds, labels=labels)

    st.markdown("### Confusion Matrix")

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    st.dataframe(cm_df, width="stretch")

    # Heatmap-style visualization
    st.markdown("### Confusion Heatmap")
    st.table(
        cm_df.style.background_gradient(cmap="Blues")
    )

    # Classification Report
    st.markdown("### Precision / Recall / F1 Score")

    report = classification_report(
        true_labels,
        preds,
        labels=labels,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(3), width="stretch")

    st.markdown("</div>", unsafe_allow_html=True)


# ===================================================
# 🤖 TAB 5 — ChatBot Assistant
# ===================================================

# ===============================
#  BANKBOT ASSISTANT TAB
# ===============================
with tab5:

    st.markdown("## 🏦 BankBot Assistant")
    st.caption("Chat with BankBot")

    # -------------------------------
    # 🆕 New Chat Button
    # -------------------------------
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("🆕 New Chat"):
            st.session_state.confirm_new_chat = True

    if st.session_state.get("confirm_new_chat"):
            st.warning("⚠️ Start a new conversation?")
            c1, c2 = st.columns(2)

            if c1.button("Yes"):
                st.session_state.chat_history = []
                st.session_state.dialogue_state = DialogueState()
                st.session_state.greeted = False
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.confirm_new_chat = False
                st.rerun()

            if c2.button("No"):
                st.session_state.confirm_new_chat = False


    if "dialogue_state" not in st.session_state:
        st.session_state.dialogue_state = DialogueState()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if not st.session_state.get("greeted", False):
        time.sleep(1)  # optional delay
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"👋 Hello {display_name}! How can I help you today?"
        })
        st.session_state.greeted = True


    manager = DialogueManager()
    classifier = IntentClassifier(MODEL_DIR)
    router = QueryRouter(classifier)

    # Render chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask banking questions...")

    if user_input:
        log_chat(
        session_id=st.session_state.session_id,
        role="user",
        message=user_input
        )
        
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        # 🧠 ROUTING DECISION
        state = st.session_state.dialogue_state

        # 🔒 Force BANKING if awaiting slot OR confirmation
        if state.awaiting_slot is not None or state.awaiting_confirmation:
            route_type = "BANKING"
            top_intent=None
        else:
            route_type, top_intent = router.route(user_input)
        
        intent = top_intent["label"] if top_intent else "general_chat"
        confidence = top_intent["score"] if top_intent else 0.0

        log_user_query(
            user_input=user_input,
            intent=intent,
            confidence=confidence
        )


        if route_type == "BANKING":
            response = manager.handle(
                user_input,
                st.session_state.dialogue_state
            )

        
            if response["type"] == "success":
                reply = response.get(
                    "message",
                    "✅ Your request has been completed."
                )
            

            elif response["type"] == "confirmed":
                reply = INTENT_STORIES[
                    st.session_state.dialogue_state.intent
                ]["response"](st.session_state.dialogue_state.slots)

                st.session_state.dialogue_state.reset()
        
            elif response["type"] == "slot_request":
                slot = response["missing"][0]

                SLOT_MESSAGES = {
                    "FROM_ACCOUNT": "🏦 Enter **sender account number**",
                    "FROM_PASSWORD": "🔐 Enter **sender account password**",
                    "TO_ACCOUNT": "🏦 Enter **receiver account number**",
                    "TO_PASSWORD": "🔐 Enter **receiver account password**",
                    "AMOUNT": "💰 Enter **amount to transfer**"
                }

                reply = SLOT_MESSAGES.get(slot, f"Please provide {slot}")


            elif response["type"] == "fallback":
                reply = (
                    "🏦 I am a banking assistant, BankBot.\n\n"
                    "I can help with:\n"
                    "- Balance enquiry\n"
                    "- Money transfer\n"
                    "- Card blocking\n"
                    "- Finding ATMs\n\n"
                    "For other information, please use a general assistant."
                )

            elif response["type"] == "confirm":
                reply = "⚠️ Are you sure you want to proceed? (yes/no)"

            elif response["type"] == "message":
                reply = response["message"]

            else:
                reply = "Sorry, something went wrong."

        else:
            # 🤖 NON-BANKING → LLM
            if looks_like_banking(user_input):
                reply = (
                    "🔒 I cannot assist with banking actions here.\n"
                    "Please use banking-specific commands."
                )
            else:
                reply = ask_llm(user_input)


        log_chat(
            session_id=st.session_state.session_id,
            role="assistant",
            message=reply
        )

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": reply
        })

        with st.chat_message("assistant"):
            st.markdown(reply)

        
# ===================================================
# 🏗️ TAB 6 — Account Management
# ===================================================
with tab6:
    st.markdown("## 🏗️ Account Management")

     # -------------------------------
    # ➕ CREATE ACCOUNT
    # -------------------------------
    st.markdown("### ➕ Create New Bank Account")
    st.caption("Create a new bank account")


    def handle_create_account():
        if not all([
            st.session_state.ca_user,
            st.session_state.ca_acc,
            st.session_state.ca_pwd
        ]):
            st.error("❌ All fields are required")
            return

        create_account(
            st.session_state.ca_user,
            st.session_state.ca_acc,
            st.session_state.ca_type,
            st.session_state.ca_balance,
            st.session_state.ca_pwd
        )

        st.success("✅ Account created successfully!")

        # ✅ SAFE RESET (callback context)
        st.session_state.ca_user = ""
        st.session_state.ca_acc = ""
        st.session_state.ca_type = "Savings"
        st.session_state.ca_balance = 0.0
        st.session_state.ca_pwd = ""
    
    with st.form("create_account_form"):
        user_name = st.text_input("Customer Name", key="ca_user")
        account_number = st.text_input("Account Number", key="ca_acc")
        account_type = st.selectbox("Account Type", ["Savings", "Current"], key="ca_type")
        initial_balance = st.number_input("Initial Balance", min_value=0.0, step=500.0, key="ca_balance")
        password = st.text_input("Account Password", type="password", key="ca_pwd")
        submit_account = st.form_submit_button(
            "Create Account",
            on_click=handle_create_account
        )    

    st.divider()

    # -------------------------------
    # ❌ DELETE ACCOUNT
    # -------------------------------
    st.markdown("### ❌ Delete Bank Account")

    def handle_delete_account():
        acc_no = st.session_state.delete_acc.strip()

        if not acc_no:
            st.error("❌ Please enter an account number")
            return

        deleted = delete_account(acc_no)

        if deleted:
            st.success("✅ Account deleted successfully")
            st.session_state.delete_acc = ""
        else:
            st.error("❌ Account not found")

            
        # ✅ SAFE RESET (callback context)
        st.session_state.delete_acc = ""

    
    st.text_input(
        "Account Number to Delete",
        key="delete_acc"
    )

    st.button(
        "Delete Account",
        on_click=handle_delete_account
    )

# ===================================================
# 🔐 TAB 7 — Admin Panel
# ===================================================
with tab7:
    admin_panel()
# ---------------------------------------------------
st.caption("🏦 BankBot NLU — Intent Classification + Entity Extraction (Streamlit)")