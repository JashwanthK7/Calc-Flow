import streamlit as st
import os
import requests
import re
from openai import OpenAI
import sys
import io
import contextlib
import numpy as np
import sympy as sp
import sqlalchemy
from sqlalchemy.exc import SQLAlchemyError
import matplotlib.pyplot as plt
import psycopg2
from urllib.parse import urlparse
import uuid
import datetime
import hashlib 

st.set_page_config(page_title="CalcFlow Tutor", page_icon="∫", layout="wide")

# --- INITIALIZE SESSION STATE ---
if "quiz_started" not in st.session_state:
    st.session_state.quiz_started = False
if "quiz_history" not in st.session_state:
    st.session_state.quiz_history = []
if "current_question" not in st.session_state:
    st.session_state.current_question = None
if "current_difficulty" not in st.session_state:
    st.session_state.current_difficulty = "Easy"
if "quiz_topic" not in st.session_state:
    st.session_state.quiz_topic = "Derivatives"
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0
if "feedback_visible" not in st.session_state:
    st.session_state.feedback_visible = False
if "last_truth" not in st.session_state:
    st.session_state.last_truth = ""
if "last_solution" not in st.session_state:
    st.session_state.last_solution = ""
if "last_correct" not in st.session_state:
    st.session_state.last_correct = False
if "last_feedback" not in st.session_state:
    st.session_state.last_feedback = ""
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "quiz_submitted_flag" not in st.session_state:
    st.session_state.quiz_submitted_flag = False
if "messages" not in st.session_state:
    st.session_state.messages = []


st.markdown(
    """
    <style>
    :root {
        --bg-main: #020617;
        --bg-card: #020617;
        --border-subtle: #1f2937;
        --text-main: #e5e7eb;
        --text-muted: #9ca3af;
        --accent: #3b82f6;
        --accent-soft: rgba(59,130,246,0.35);
    }

    .stApp {
        background:
            radial-gradient(circle at 0% 0%, rgba(56,189,248,0.16) 0, transparent 40%),
            radial-gradient(circle at 100% 0%, rgba(129,140,248,0.16) 0, transparent 45%),
            radial-gradient(circle at 50% 120%, rgba(59,130,246,0.18) 0, transparent 55%),
            #020617;
    }

    .block-container {
        padding-top: 3.2rem;   /* extra top padding so hero is not cut */
        padding-bottom: 2rem;
        max-width: 1100px;
        color: var(--text-main);
    }

    /* Hero card */
    .cf-hero {
        background: rgba(2,6,23,0.96);
        padding: 1.3rem 1.5rem;
        border-radius: 1rem;
        border: 1px solid var(--border-subtle);
        box-shadow: 0 22px 55px rgba(0,0,0,0.75);
    }
    .cf-hero-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        color: var(--text-main);
    }
    .cf-hero-sub {
        font-size: 0.95rem;
        color: var(--text-muted);
        margin: 0;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-main) !important;
    }

    /* Text */
    p, li, span, div {
        color: var(--text-main);
    }

    /* Under-button captions */
    .cf-nav-caption {
        font-size: 0.85rem;
        color: var(--text-muted);
        margin-top: 0.15rem;
    }

    /* Mode avatars above buttons */
    .cf-mode-avatar {
        width: 60px;
        height: 60px;
        border-radius: 999px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 0.4rem auto;
        background: radial-gradient(circle at 30% 0%, #f97316 0, #ea580c 30%, #020617 100%);
        box-shadow: 0 12px 35px rgba(249,115,22,0.55);
        font-size: 1.5rem;
    }
    .cf-mode-avatar.quiz {
        background: radial-gradient(circle at 30% 0%, #22c55e 0, #16a34a 30%, #020617 100%);
        box-shadow: 0 12px 35px rgba(34,197,94,0.55);
    }

    /* Global buttons */
    .stButton>button {
        border-radius: 999px;
        padding: 0.85rem 1.25rem;
        font-size: 1.0rem;
        font-weight: 500;
        border: 1px solid var(--border-subtle);
        background: linear-gradient(135deg, #020617 0%, #0f172a 40%, #111827 100%);
        color: var(--text-main);
        transition: all 0.18s ease-in-out;
        box-shadow: 0 14px 40px rgba(0,0,0,0.75);
    }
    .stButton>button:hover {
        border-color: var(--accent);
        box-shadow: 0 0 0 1px var(--accent-soft), 0 22px 60px rgba(37,99,235,0.55);
        transform: translateY(-1px);
    }
    
    /* Specific styling for History buttons */
    div[data-testid="stSidebar"] .stButton>button {
        text-align: left;
        border: 1px solid transparent;
        background: transparent;
        box-shadow: none;
        padding: 0.5rem;
        color: var(--text-muted);
        font-weight: 400;
        font-size: 0.9rem;
        width: 100%;
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
        display: block;
    }
    div[data-testid="stSidebar"] .stButton>button:hover {
        background: rgba(255,255,255,0.05);
        color: var(--text-main);
        border-color: var(--border-subtle);
        transform: none;
        box-shadow: none;
    }
    /* Active chat highlight */
    div[data-testid="stSidebar"] .stButton>button:focus {
        border-color: var(--accent);
        color: var(--accent);
    }

    /* Chat messages */
    .stChatMessage {
        color: var(--text-main) !important;
    }

    /* KaTeX math color */
    .katex, .katex * {
        color: var(--text-main) !important;
    }

    /* Alert boxes */
    .stAlert {
        background-color: #020617 !important;
        border-radius: 0.75rem !important;
        border: 1px solid var(--border-subtle) !important;
    }

    /* Progress bar text */
    .stProgress p {
        color: var(--text-muted) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- CONFIGURATION ---

# Environment Variables
HOST_URL = os.environ.get("HOST", "localhost")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://10.230.100.240:17020")

# --- DATABASE CONNECTION BUILDER ---
DB_HOST = os.environ.get("POSTGRES_HOST", "postgres")
DB_USER = os.environ.get("POSTGRES_USER", "postgres")
DB_PASS = os.environ.get("POSTGRES_PASSWORD", os.environ.get("SERVICE_PASSWORD"))
DB_NAME = os.environ.get("POSTGRES_DB", "tutor_app")
DB_PORT = os.environ.get("POSTGRES_PORT", "5432")

# Construct the connection URL (DSN format for psycopg2)
if os.environ.get("DATABASE_PORT_ONE") and "host=" in os.environ.get("DATABASE_PORT_ONE"):
    # Fallback for local setup_env.sh usage
    DATABASE_URL = os.environ.get("DATABASE_PORT_ONE")
else:
    # Docker / Production usage
    DATABASE_URL = f"host={DB_HOST} port={DB_PORT} user={DB_USER} password={DB_PASS} dbname={DB_NAME}"

# AI Client Setup
if OLLAMA_BASE_URL.endswith("/v1"):
    API_BASE = OLLAMA_BASE_URL.replace("/v1", "")
    CLIENT_URL = OLLAMA_BASE_URL
else:
    API_BASE = OLLAMA_BASE_URL
    CLIENT_URL = f"{OLLAMA_BASE_URL}/v1"

@st.cache_resource
def get_connection(db_url):
    engine = sqlalchemy.create_engine(db_url)
    # Test the connection immediately to ensure credentials are correct
    with engine.connect() as conn:
        pass 
    return engine
    
# --- DATABASE SETUP ---
def init_db():
    """Initializes the PostgreSQL database table if it doesn't exist."""
    if not DB_PASS:
        st.warning("⚠️ Database password not found. DB features may be disabled.")
        return

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()

        create_table_query = """
        CREATE TABLE IF NOT EXISTS chat_logs (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(50),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            role VARCHAR(20),
            content TEXT,
            model_used VARCHAR(50)
        );
        """
        cur.execute(create_table_query)

        try:
            cur.execute("ALTER TABLE chat_logs ADD COLUMN IF NOT EXISTS user_id INTEGER;")
            conn.commit()
        except Exception:
            conn.rollback() # Ignore if already exists or fails gracefully

        create_users_table = """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(120),
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cur.execute(create_users_table)

        # NEW: quiz_logs for granular stats (per question)
        create_quiz_logs_table = """
        CREATE TABLE IF NOT EXISTS quiz_logs (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            correct BOOLEAN NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cur.execute(create_quiz_logs_table)

        # NEW: completed_quizzes for dashboard counts (full session)
        create_completed_quizzes_table = """
        CREATE TABLE IF NOT EXISTS completed_quizzes (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            score INTEGER,
            total_questions INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cur.execute(create_completed_quizzes_table)

        # NEW: feedback_logs for reinforcement learning data
        create_feedback_table = """
        CREATE TABLE IF NOT EXISTS feedback_logs (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(50),
            user_id INTEGER,
            message_index INTEGER,
            feedback_type VARCHAR(20),
            prompt TEXT,
            response TEXT,
            model_used VARCHAR(50),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cur.execute(create_feedback_table)

        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"⚠️ Database Connection Error: {e}")


def log_to_db(role, content, model="unknown"):
    """Logs a message to the PostgreSQL database."""
    if not DB_PASS:
        return

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    user_id = st.session_state.user['id'] if st.session_state.user else None

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        insert_query = """
        INSERT INTO chat_logs (session_id, user_id, role, content, model_used)
        VALUES (%s, %s, %s, %s, %s)
        """
        cur.execute(insert_query, (st.session_state.session_id, user_id, role, content, model))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Logging failed: {e}")

def log_feedback_db(session_id, user_id, msg_index, feedback_type, prompt, response, model):
    """Logs user feedback (thumbs up/down) to the database."""
    if not DB_PASS:
        return

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        insert_query = """
        INSERT INTO feedback_logs (session_id, user_id, message_index, feedback_type, prompt, response, model_used)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cur.execute(insert_query, (session_id, user_id, msg_index, feedback_type, prompt, response, model))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Feedback logging failed: {e}")


# FORCE INIT DB ON EVERY RUN to ensure new tables are created
# Removed the `if "db_initialized" ...` check to fix schema update issues
init_db() 
st.session_state.db_initialized = True

def get_db_connection():
    """Small helper so we don't repeat psycopg2.connect everywhere."""
    return psycopg2.connect(DATABASE_URL)


# ---------- PASSWORD HASHING (simple but better than plain text) ----------
def hash_password(password: str) -> str:
    """Return salt$hash using SHA256."""
    salt = os.urandom(16).hex()
    hashed = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return f"{salt}${hashed}"


def verify_password(password: str, stored: str) -> bool:
    """Compare plain password with stored salt$hash."""
    try:
        salt, hashed = stored.split("$", 1)
    except ValueError:
        return False
    check = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return check == hashed


# ---------- USER CRUD ----------
def create_user(username: str, email: str, password: str):
    """Create a new user. Returns (success, message)."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Check if username already exists
        cur.execute("SELECT id FROM users WHERE username = %s;", (username,))
        if cur.fetchone():
            cur.close()
            conn.close()
            return False, "Username already taken."

        pwd_hash = hash_password(password)
        cur.execute(
            """
            INSERT INTO users (username, email, password_hash)
            VALUES (%s, %s, %s)
            RETURNING id;
            """,
            (username, email, pwd_hash),
        )
        user_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return True, f"User created with id {user_id}"
    except Exception as e:
        return False, f"Error creating user: {e}"


def get_user_by_username(username: str):
    """Fetch a user row by username. Returns dict or None."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, username, email, password_hash FROM users WHERE username = %s;",
            (username,),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            return {
                "id": row[0],
                "username": row[1],
                "email": row[2],
                "password_hash": row[3],
            }
        return None
    except Exception:
        return None


def authenticate_user(username: str, password: str):
    """Return (success, user_dict or None, message)."""
    user = get_user_by_username(username)
    if not user:
        return False, None, "User not found."
    if not verify_password(password, user["password_hash"]):
        return False, None, "Incorrect password."
    return True, user, "Login successful."

def get_user_quiz_stats(user_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("SELECT COUNT(*) FROM completed_quizzes WHERE user_id = %s;", (user_id,))
        total_quizzes = cur.fetchone()[0] or 0

        # We force floating point division to be safe, though most SQL does this automatically for these types
        cur.execute("""
            SELECT AVG(CAST(score AS FLOAT) / NULLIF(total_questions, 0))
            FROM completed_quizzes
            WHERE user_id = %s;
        """, (user_id,))
        avg_score_val = cur.fetchone()[0]
        
        cur.close()
        conn.close()
        
        avg_score = round(avg_score_val * 100, 2) if avg_score_val is not None else 0
        return total_quizzes, avg_score
    except Exception as e:
        # st.error(f"DB Error: {e}") # Uncomment to debug
        return 0, 0


def get_recent_questions(user_id, limit=5):
    """Fetch recent activity for the user using user_id instead of session_id."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT content, timestamp
            FROM chat_logs
            WHERE user_id = %s AND role = 'user'
            ORDER BY timestamp DESC
            LIMIT %s;
        """, (user_id, limit))
        
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception:
        return []

def get_user_sessions(user_id):
    """Get all session IDs and their first user message (title) for a user."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        # Postgres specific distinct on logic to get the FIRST message of each session
        cur.execute("""
            SELECT DISTINCT ON (session_id) session_id, content, timestamp
            FROM chat_logs
            WHERE user_id = %s AND role = 'user'
            ORDER BY session_id, timestamp ASC;
        """, (user_id,))
        
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        # Sort by timestamp descending (newest sessions first)
        rows.sort(key=lambda x: x[2], reverse=True)
        return [{"id": r[0], "title": r[1][:40] + "..." if len(r[1]) > 40 else r[1]} for r in rows]
    except Exception as e:
        print(f"Error fetching sessions: {e}")
        return []

def load_chat_history(user_id, session_id):
    """Load chat messages for a specific session."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT role, content
            FROM chat_logs
            WHERE user_id = %s AND session_id = %s AND role IN ('user', 'assistant')
            ORDER BY timestamp ASC;
        """, (user_id, session_id))
        
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [{"role": row[0], "content": row[1]} for row in rows]
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return []

    
# --- PROMPTS ---
TUTOR_PROMPT = """
You are CalcFlow, an expert AI Calculus Tutor.

SCOPE (VERY IMPORTANT):
- You ONLY help with mathematics, especially calculus topics:
  limits, continuity, derivatives, integrals, optimization, series,
  differential equations, related rates, etc.
- If the user asks about ANYTHING non-math (cooking, travel, gossip, life,
  coding, health, etc.), you MUST NOT answer that request.
- Instead, politely say something like:
  "I'm CalcFlow, a calculus tutor. I can only help with math and calculus questions.
   Try turning your question into a math problem."

STYLE:
- Explain in clear, simple English.
- Use only simple LaTeX math inside $...$ or $$...$$.
- Avoid \\begin / \\end environments and fancy macros (no \\overset, \\underset,
  \\text, \\underbrace, etc.).
- Present the solution in small, numbered steps.
- Give the final answer near the end, then ask a short follow-up question.

Example of allowed math: $x^2$, $\\frac{3}{4}$, $\\lim_{x\\to 2} f(x)$, $\\sqrt{x}$.
"""

QUIZ_PROMPT = """
You are a Calculus Quiz Master.

Your task:
- Generate ONE single Calculus I problem for the requested TOPIC and DIFFICULTY.
- **ALWAYS use the variable 'x' (not t, theta, or y).**
- The problem MUST be solvable with a real, finite answer.
- Use clean numbers (integers or simple fractions).

CRITICAL RESTRICTIONS (VERY IMPORTANT):
- DO NOT show any steps, hints, reasoning, or explanations.
- DO NOT show or mention the final answer.
- Just write the problem statement itself, in LaTeX if needed.

Output format:
- A short problem statement only.
"""

SOLVER_PROMPT = """
You are a Python Coding Assistant for Calculus.
Write a Python script using the `sympy` library to solve the following problem.
- Define variables using `x = sympy.symbols('x')`
- Print ONLY the final result.
- Do NOT explain the code. Just write the code block.
"""

JUDGE_PROMPT = """
You are a Calculus Grader.

1. The Student Answer is: "{student_answer}"
2. The True Calculated Answer (from Python) is: "{python_result}"

Task:
- Decide if the student's answer is mathematically equivalent to the true answer.
- Treat equivalent forms as CORRECT (e.g., "1/2" == "0.5", "x^2" == "x**2", "ln(x)" == "log(x)").
- Ignore trivial formatting differences, whitespace, or ordering of factors.
- Only mark INCORRECT if the expressions are genuinely different.

Output format (strict):
[[CORRECT]] your short explanation
or
[[INCORRECT]] your short explanation
"""

SOLUTION_PROMPT = """
You are a calculus tutor.

RULES:
- Use only simple LaTeX math (powers, fractions, limits, square roots).
- Use $...$ for inline math and $$...$$ for display math.
- Never put more than TWO dollar signs in a row (no $$$ or $$$$).
- Do NOT use any LaTeX environments like \\begin or \\end.
- Do NOT use commands such as \\overset, \\underset, \\underbrace, \\phantom, \\Large, \\small, \\text, etc.
- Do NOT use multi-line align blocks.
- Do NOT include \\displaystyle.

Your job:
1. Break the solution into clear steps in English.
2. Put math inside $ ... $ for inline math or $$ ... $$ for display math.
3. Make each step short and readable.

Problem:
{problem}

Correct final answer:
{python_result}

Now produce a clean, step-by-step solution.
"""

def get_available_models():
    try:
        response = requests.get(f"{API_BASE}/api/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            return [model['name'] for model in data['models']]
    except Exception:
        pass
    return ["llama3.1:latest", "mistral:latest", "llama2:latest"]

def get_best_model():
    try:
        response = requests.get(f"{API_BASE}/api/tags", timeout=2)
        if response.status_code == 200:
            models = [m['name'] for m in response.json().get('models', [])]
            for preferred in ["llama3:latest", "llama3.1:latest", "mistral:latest", "llama2:latest"]:
                if preferred in models:
                    return preferred
            if models:
                return models[0]
    except Exception:
        pass
    return "llama3:latest"

def format_math(text: str) -> str:
    """
    Clean up LLM output so Streamlit/KaTeX render math instead of raw LaTeX code.
    We aggressively strip unsupported environments and fix common mistakes.
    """
    if not text:
        return ""

    text = re.sub(r"```[a-zA-Z]*", "", text)
    text = text.replace("```", "")

    patterns_to_remove = [
        r"\\begin\{[^}]+\}", r"\\end\{[^}]+\}",
        r"\\overset\{[^}]*\}\{[^}]*\}", r"\\underset\{[^}]*\}\{[^}]*\}",
        r"\\mathbf\{[^}]*\}", r"\\mathrm\{[^}]*\}",
        r"\\Large", r"\\large", r"\\small", r"\\tiny",
        r"\\underbrace\{[^}]*\}\{[^}]*\}",
        r"\\phantom\{[^}]*\}",
    ]
    for p in patterns_to_remove:
        text = re.sub(p, "", text)

    # Remove \text{...} but keep the contents
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)

    # Convert \( \) and \[ \] to $ and $$ for KaTeX
    text = text.replace(r"\(", "$").replace(r"\)", "$")
    text = text.replace(r"\[", "$$").replace(r"\]", "$$")

    # Fix accidental triple-or-more $ like $$$\lim -> $$\lim
    text = re.sub(r"\${3,}", "$$", text)
    text = re.sub(r"(?<!\\)frac\{", r"\\frac{", text)
    text = re.sub(r"(?<!\\)sqrt\{", r"\\sqrt{", text)
    text = re.sub(r"(?<!\\)sin\(", r"\\sin(", text)
    text = re.sub(r"(?<!\\)cos\(", r"\\cos(", text)
    text = re.sub(r"(?<!\\)tan\(", r"\\tan(", text)
    text = re.sub(r"(?<!\\)log\(", r"\\log(", text)

    # Normalize display math spacing
    text = text.replace("$$\n", "$$")
    text = text.replace("\n$$", "$$")

    # Collapse too many blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_title_text(text):
    text = text.replace("$$", "").replace("$", "").replace(r"\[", "").replace(r"\]", "")
    text = text.replace("{", "").replace("}", "").replace("[", "").replace("]", "")
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:60]

def maybe_generate_plot(prompt: str):
    pattern = r"y\s*=\s*(.+?)\s*(?:for|from)\s*x\s*=?\s*([-\d\.]+)\s*(?:to|-)\s*([-\d\.]+)"
    match = re.search(pattern, prompt, re.IGNORECASE)
    if not match:
        return None

    expr_str, x_min_str, x_max_str = match.groups()

    try:
        x = sp.symbols("x")
        expr = sp.sympify(expr_str, {"x": x, "pi": sp.pi, "e": sp.E})

        x_min = float(x_min_str)
        x_max = float(x_max_str)
        if x_min >= x_max:
            raise ValueError("x_min must be less than x_max")

        xs = np.linspace(x_min, x_max, 400)
        f = sp.lambdify(x, expr, "numpy")

        ys = f(xs)
        ys = np.array(ys, dtype=float)

        if not np.any(np.isfinite(ys)):
            st.warning("Could not generate numeric values for this function.")
            return None

        fig, ax = plt.subplots()
        ax.plot(xs, ys)
        ax.set_title(f"Graph of y = {expr_str.strip()} on [{x_min}, {x_max}]")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)

        return fig
    except Exception as e:
        st.warning(f"Could not generate plot: {e}")
        return None

def get_ai_response(messages, model, stream=True):
    client = OpenAI(base_url=CLIENT_URL, api_key="ollama")
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=stream
    )
    return completion

def adjust_difficulty(current_diff, was_correct):
    levels = ["Easy", "Medium", "Hard"]
    idx = levels.index(current_diff)
    if was_correct:
        return levels[min(idx + 1, 2)]
    else:
        return levels[max(idx - 1, 0)]

def execute_math_code(code_str):
    output_buffer = io.StringIO()
    clean_code = code_str.replace("```python", "").replace("```", "").strip()

    import sympy
    safe_globals = {
        "sympy": sympy,
        "sp": sympy,
        "print": print
    }
    
    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(clean_code, safe_globals)
        return output_buffer.getvalue().strip()
    except Exception as e:
        return f"Error executing code: {e}"

def sanitize_quiz_question(raw: str) -> str:
    """
    If the model tries to include 'Step 1', 'Solution', 'Final answer', etc.,
    cut the text there so the student only sees the question, not the solution.
    """
    if not raw:
        return ""
    lower = raw.lower()
    cut = len(raw)
    stops = ["step 1", "step one", "solution", "final answer", "answer:", "the final answer"]
    for s in stops:
        idx = lower.find(s)
        if idx != -1:
            cut = min(cut, idx)
    raw = raw[:cut]
    return raw.strip()
def interpret_graph_request(request: str):
    """
    Turn a natural language request like:
        'plot the derivative of sin(x)'
        'graph second derivative of x**3'
        'plot sin(x) from -5 to 5'
    into: (expr_to_plot, label, x_min, x_max), error
    """
    if not request:
        return None, "Empty request."

    text = request.strip()
    lower = text.lower()

    # Default x range if user doesn't specify
    x_min, x_max = -10.0, 10.0

    # Try to detect 'from a to b' or 'between a and b'
    m = re.search(r'(?:from|between)\s*([-\d\.]+)\s*(?:to|and)\s*([-\d\.]+)', lower)
    if m:
        try:
            x_min = float(m.group(1))
            x_max = float(m.group(2))
        except Exception:
            pass  # keep defaults if parse fails

    # Remove filler words
    cleaned = lower
    for w in ["plot", "graph", "show", "the", "a", "an", "please"]:
        cleaned = cleaned.replace(w, "")
    cleaned = cleaned.strip()

    # Detect derivative order
    order = 0
    if "second derivative" in cleaned or "2nd derivative" in cleaned:
        order = 2
        cleaned = cleaned.replace("second derivative of", "")
        cleaned = cleaned.replace("2nd derivative of", "")
        cleaned = cleaned.replace("second derivative", "")
        cleaned = cleaned.replace("2nd derivative", "")
    elif "first derivative" in cleaned or "1st derivative" in cleaned or "derivative" in cleaned:
        order = 1
        cleaned = cleaned.replace("first derivative of", "")
        cleaned = cleaned.replace("1st derivative of", "")
        cleaned = cleaned.replace("derivative of", "")
        cleaned = cleaned.replace("first derivative", "")
        cleaned = cleaned.replace("1st derivative", "")
        cleaned = cleaned.replace("derivative", "")

    cleaned = cleaned.strip()

    # If we still have ' of ' left, grab whatever is after it
    if " of " in cleaned:
        cleaned = cleaned.split(" of ", 1)[-1].strip()

    # If nothing useful left, just fall back to original text
    if not cleaned:
        cleaned = text

    # Support x^2 style
    cleaned = cleaned.replace("^", "**")

    x = sp.symbols("x")
    try:
        expr_base = sp.sympify(cleaned, {
            "x": x,
            "pi": sp.pi,
            "e": sp.E,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "exp": sp.exp,
            "log": sp.log,
            "sqrt": sp.sqrt,
            "abs": sp.Abs,
        })
    except Exception as e:
        return None, f"Could not understand the expression: {e}"

    expr_to_plot = expr_base
    label = f"f(x) = {cleaned}"

    if order == 1:
        expr_to_plot = sp.diff(expr_base, x)
        label = f"f'(x) (derivative of {cleaned})"
    elif order == 2:
        expr_to_plot = sp.diff(expr_base, (x, 2))
        label = f"f''(x) (second derivative of {cleaned})"

    return (expr_to_plot, label, x_min, x_max), None

def clean_function_input(func_str: str) -> str:
    """
    Normalize user input so SymPy can parse it.
    Handles:
    - Unicode minus signs
    - Removing 'y=' or 'f(x)=' prefixes
    - Replacing ^ with **
    - Normalizing names (cosec -> csc, arcsin -> asin, ln -> log)
    - Handling implied multiplication (2x -> 2*x, secx -> sec(x))
    """
    s = func_str.strip()

    # 1. Basic Cleanup
    s = s.replace("−", "-").replace("–", "-")
    s = re.sub(r"^[yf]\s*\(?x?\)?\s*=", "", s, flags=re.IGNORECASE)
    s = s.replace("^", "**")

    # 2. Normalize Synonyms (SymPy prefers csc, asin, log)
    s = re.sub(r"\bcosec\b", "csc", s, flags=re.IGNORECASE)
    s = re.sub(r"\barcsin\b", "asin", s, flags=re.IGNORECASE)
    s = re.sub(r"\barccos\b", "acos", s, flags=re.IGNORECASE)
    s = re.sub(r"\barctan\b", "atan", s, flags=re.IGNORECASE)
    s = re.sub(r"\barcsec\b", "asec", s, flags=re.IGNORECASE)
    s = re.sub(r"\barccsc\b", "acsc", s, flags=re.IGNORECASE)
    s = re.sub(r"\barccot\b", "acot", s, flags=re.IGNORECASE)
    s = re.sub(r"\bln\b", "log", s, flags=re.IGNORECASE)

    # 3. Insert explicit multiplication: 2x -> 2*x, x(x+1) -> x*(x+1)
    s = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", s)
    s = re.sub(r"(\d)\(", r"\1*(", s)
    s = re.sub(r"\)([a-zA-Z])", r")*\1", s)

    # 4. Handle implied parentheses for functions (e.g., "secx" -> "sec(x)")
    # List of supported functions to check for
    funcs = [
        "sin", "cos", "tan", "sec", "csc", "cot",
        "asin", "acos", "atan", "asec", "acsc", "acot",
        "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
        "exp", "log", "sqrt", "abs"
    ]
    func_pattern = "|".join(funcs)
    
    # Match "sin x" -> "sin(x)"
    s = re.sub(fr"\b({func_pattern})\s+([a-zA-Z][a-zA-Z0-9_]*)", r"\1(\2)", s, flags=re.IGNORECASE)
    # Match "sinx" -> "sin(x)" (Lookahead ensures we don't double-capture)
    s = re.sub(fr"\b({func_pattern})(?!\s*\()([a-zA-Z][a-zA-Z0-9_]*)", r"\1(\2)", s, flags=re.IGNORECASE)

    return s.strip()

def check_answer_strict(user_str, truth_str):
    """
    Strictly verifies if user_str is mathematically equivalent to truth_str
    using symbolic subtraction. Returns (is_correct, feedback).
    """
    if not user_str or not truth_str:
        return False, "Missing answer."

    # 1. Clean inputs
    u_clean = clean_function_input(user_str)
    t_clean = clean_function_input(truth_str)

    # 2. Handle "Error" cases from the engine
    if "error" in t_clean.lower():
        return False, "The math engine encountered an error, so we cannot verify strictly."

    try:
        # 3. Define Context (same as Graph Lab)
        x = sp.symbols('x')
        C = sp.symbols('C') # For integrals
        
        safe_math = {
            "x": x, "C": C,
            "pi": sp.pi, "e": sp.E,
            "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
            "sec": sp.sec, "csc": sp.csc, "cot": sp.cot,
            "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
            "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs,
        }

        # 4. Parse
        user_expr = sp.sympify(u_clean, locals=safe_math)
        true_expr = sp.sympify(t_clean, locals=safe_math)

        # 5. The "Subtraction Test"
        # If (A - B) simplifies to 0, they are equal.
        diff = sp.simplify(user_expr - true_expr)
        
        if diff == 0:
            return True, "Correct! Exact match."
        
        # Double check: sometimes simplify(0.5 - 1/2) is tricky with floats
        # Try evaluating at a random point as a fallback check
        val = 1.2345
        user_val = user_expr.subs(x, val).evalf()
        true_val = true_expr.subs(x, val).evalf()
        
        if abs(user_val - true_val) < 1e-6:
             return True, "Correct! (Verified numerically)"

        return False, f"Incorrect. expected {t_clean}, got {u_clean}."

    except Exception as e:
        return False, f"Could not verify math: {e}"

FUNNY_QUOTES = [
    "Calculus builds character. And tears. Mostly tears.",
    "Derivatives are just slopes trying to feel important.",
    "Math is 90% staring at the problem and 10% panic.",
    "You + CalcFlow = unstoppable (unless it's integrals).",
    "If at first you don’t succeed, try taking the derivative.",
]


# --- MODES / VIEWS ---
MODE_HOME = "HOME"
MODE_CHAT = "CHAT"
MODE_QUIZ = "QUIZ"
MODE_GRAPH = "GRAPH"
MODE_HANDBOOK = "HANDBOOK"
MODE_DASHBOARD = "DASHBOARD"

if "active_model" not in st.session_state:
    st.session_state.active_model = get_best_model()

if "mode" not in st.session_state:
    st.session_state.mode = MODE_HOME  # start on home screen

# ---------- AUTH GATE (Login / Signup) ----------

def auth_screen():
    st.title("The Hall of Integrals")
    st.caption("Login Here")

    # 2. CHECK LOGIN STATUS
    # If the engine is already in session_state, skip the login form.
    if "db_engine" in st.session_state:
        st.success("✅ You are already connected to the database.")
        if st.button("Logout"):
            # Clear state and cache to logout
            st.session_state.pop("db_engine", None)
            st.cache_resource.clear()
            st.rerun()
        return

    # --- Initialize the tabs here ---
    tab_login, tab_signup = st.tabs(["Login", "Sign Up"]) 

    # ---- LOGIN TAB ----
    with tab_login:
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Log In"):
            if not DATABASE_URL:
                st.error("Database not available for login.")
            elif not login_username or not login_password:
                st.warning("Please enter both username and password.")
            else:
                ok, user, msg = authenticate_user(login_username, login_password)
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.user = user
                    # START NEW SESSION BY DEFAULT
                    st.session_state.session_id = str(uuid.uuid4())
                    st.session_state.messages = []
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error(msg)

    # ---- SIGNUP TAB ----
    with tab_signup:
        new_username = st.text_input("Choose a username", key="signup_username")
        new_email = st.text_input("Email (optional)", key="signup_email")
        new_pwd = st.text_input("Password", type="password", key="signup_password")
        new_pwd2 = st.text_input("Confirm password", type="password", key="signup_password2")

        if st.button("Create Account"):
            if not DATABASE_URL:
                st.error("Database not available for signup.")
            elif not new_username or not new_pwd or not new_pwd2:
                st.warning("Username and both password fields are required.")
            elif new_pwd != new_pwd2:
                st.warning("Passwords do not match.")
            else:
                ok, msg = create_user(new_username, new_email, new_pwd)
                if ok:
                    st.success("Account created! You can now log in from the Login tab.")
                else:
                    st.error(msg)

# If not logged in, show auth screen and STOP
if not st.session_state.logged_in:
    auth_screen()
    st.stop()

# --- HERO + TOP NAV ---
st.markdown(
    """
    <div class="cf-hero">
        <div class="cf-hero-title">∫ CalcFlow Tutor</div>
        <p class="cf-hero-sub">
            Interactive calculus tutor and adaptive quiz engine in one place.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(f"**Status:** Online &nbsp;&nbsp;|&nbsp;&nbsp; **Node:** `{HOST_URL}`")

st.markdown("### Choose how you want to learn today:")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("<div class='cf-mode-avatar'>🤖</div>", unsafe_allow_html=True)
    if st.button("Tutor Chat", use_container_width=True):
        st.session_state.mode = MODE_CHAT
        st.rerun()
    st.markdown(
        "<div class='cf-nav-caption'>Concept-by-concept explanations.</div>",
        unsafe_allow_html=True,
    )

with c2:
    st.markdown("<div class='cf-mode-avatar quiz'>🎯</div>", unsafe_allow_html=True)
    if st.button("Adaptive Quiz", use_container_width=True):
        st.session_state.mode = MODE_QUIZ
        st.rerun()
    st.markdown(
        "<div class='cf-nav-caption'>10-question adaptive practice.</div>",
        unsafe_allow_html=True,
    )

with c3:
    # reuse avatar style, but different emoji
    st.markdown("<div class='cf-mode-avatar quiz'>📈</div>", unsafe_allow_html=True)
    if st.button("Graph Lab", use_container_width=True):
        st.session_state.mode = MODE_GRAPH
        st.rerun()
    st.markdown(
        "<div class='cf-nav-caption'>Visualize functions and curves.</div>",
        unsafe_allow_html=True,
    )


st.write("")  # small gap

mode = st.session_state.mode

with st.sidebar:
    st.header("Menu")
    if mode == MODE_HOME:
        st.caption("Current mode: Home")
    elif mode == MODE_CHAT:
        st.caption("Current mode: Tutor Chat")
        
        # --- CHAT HISTORY SIDEBAR ---
        if st.session_state.logged_in:
            st.divider()
            st.subheader("Chat History")
            
            # New Chat Button
            if st.button("+ New Chat", use_container_width=True):
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.messages = []
                st.rerun()
                
            st.write("") # Gap

            # Load Sessions
            sessions = get_user_sessions(st.session_state.user['id'])
            
            for s in sessions:
                # Highlight active session
                is_active = (s["id"] == st.session_state.session_id)
                # We use a trick to style active vs inactive if needed, but for now simple buttons
                label = s["title"] if s["title"] else "New Chat"
                if is_active:
                    label = f"🔵 {label}"
                
                if st.button(label, key=s["id"]):
                    st.session_state.session_id = s["id"]
                    st.session_state.messages = load_chat_history(st.session_state.user['id'], s["id"])
                    st.rerun()


    elif mode == MODE_QUIZ:
        st.caption("Current mode: Adaptive Quiz")
    elif mode == MODE_GRAPH:
        st.caption("Current mode: Graph Lab")
    elif mode == MODE_HANDBOOK:
        st.caption("Current mode: Handbook")
    elif mode == MODE_DASHBOARD:
        st.caption("Current mode: My Dashboard")

    st.divider()
    if st.button("Home"):
        st.session_state.mode = MODE_HOME
        st.rerun()

    # show Dashboard button only for logged-in users
    if st.session_state.logged_in and st.session_state.user:
        if st.button("My Dashboard"):
            st.session_state.mode = MODE_DASHBOARD
            st.rerun()

    if st.button("Formula Handbook"):
        st.session_state.mode = MODE_HANDBOOK
        st.rerun()

    if st.button("Reset Session"):
        st.session_state.messages = []
        st.session_state.quiz_history = []
        st.session_state.current_question = None
        st.session_state.quiz_started = False
        st.session_state.quiz_score = 0
        st.session_state.current_difficulty = "Easy"
        st.session_state.mode = MODE_HOME
        st.rerun()
        
    st.divider()
    # ---------- LOGIN STATUS BLOCK ----------
    if st.session_state.logged_in and st.session_state.user:
        st.caption(f"Logged in as **{st.session_state.user['username']}**")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.session_state.messages = []
            st.session_state.quiz_history = []
            st.session_state.mode = MODE_HOME
            st.rerun()
    else:
        st.caption("Not logged in")

    st.caption(f"Brain: {st.session_state.active_model}")
    if DATABASE_URL:
        st.caption("✅ DB Connected")
    else:
        st.caption("⚠️ DB Disconnected")

# --- HOME MODE ---
if mode == MODE_HOME:
    st.info("Choose your destiny. I promise not to judge your derivatives.")

elif mode == MODE_DASHBOARD:
    st.header("My Dashboard")

    if not st.session_state.logged_in:
        st.warning("Please log in to view your dashboard.")
        st.stop()

    user = st.session_state.user
    username = user["username"]
    user_id = user["id"]

    # Funny Quote
    import random
    quote = random.choice(FUNNY_QUOTES)

    st.markdown(f"""
        <div style="padding:1.2rem; border-radius:10px; 
                    background:rgba(255,255,255,0.05); 
                    border:1px solid #1f2937;">
            <h3>Welcome back, {username}! </h3>
            <p><i>{quote}</i></p>
        </div>
    """, unsafe_allow_html=True)

    st.write("")

    # Fetch quiz stats
    total_quizzes, avg_score = get_user_quiz_stats(user_id)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Quizzes Taken", total_quizzes)
    with col2:
        st.metric("Average Score (%)", avg_score)

    st.write("---")

    st.subheader("Recent Activity")
    recent = get_recent_questions(user_id)

    if not recent:
        st.info("No recent activity yet. Go learn something cool!")
    else:
        for content, ts in recent:
            st.markdown(f"""
                **{ts.strftime('%Y-%m-%d %H:%M:%S')}** {content}
                ---
            """)

# --- TUTOR CHAT MODE --- 
elif mode == MODE_CHAT:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):

                if msg["role"] == "assistant" and msg.get("plot_prompt"):
                    fig_hist = maybe_generate_plot(msg["plot_prompt"])
                    if fig_hist is not None:
                        st.pyplot(fig_hist, use_container_width=True)

                st.markdown(format_math(msg["content"]))

                if msg["role"] == "assistant":
                    col1, col2, col3 = st.columns([1, 1, 8])
                    
                    # Prepare feedback data
                    user_id = st.session_state.user['id'] if st.session_state.user else None
                    prompt_text = st.session_state.messages[idx-1]["content"] if idx > 0 else ""
                    response_text = msg["content"]
                    
                    with col1:
                        if st.button("👍", key=f"up_{idx}"):
                            log_feedback_db(
                                st.session_state.session_id, 
                                user_id, 
                                idx, 
                                "positive", 
                                prompt_text, 
                                response_text, 
                                st.session_state.active_model
                            )
                            st.toast("Feedback recorded: Helpful!")
                    with col2:
                        if st.button("👎", key=f"down_{idx}"):
                            log_feedback_db(
                                st.session_state.session_id, 
                                user_id, 
                                idx, 
                                "negative", 
                                prompt_text, 
                                response_text, 
                                st.session_state.active_model
                            )
                            st.toast("Feedback recorded: Needs Improvement.")

    if prompt := st.chat_input("Ask a calculus question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        log_to_db("user", prompt, st.session_state.active_model)

        with st.chat_message("user"):
            st.markdown(prompt)

        ai_messages = [{"role": "system", "content": TUTOR_PROMPT}] + st.session_state.messages

        fig = None
        with st.chat_message("assistant"):
            fig = maybe_generate_plot(prompt)
            if fig is not None:
                st.markdown("Graph of your function:")
                st.pyplot(fig, use_container_width=True)

            msg_placeholder = st.empty()
            full_response = ""
            try:
                stream = get_ai_response(ai_messages, st.session_state.active_model)
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        msg_placeholder.markdown(format_math(full_response) + "▌")
                msg_placeholder.markdown(format_math(full_response))
            except Exception as e:
                st.error(f"AI Error: {e}")
                full_response = "Error connecting to AI."

        assistant_msg = {"role": "assistant", "content": full_response}
        if fig is not None:
            assistant_msg["plot_prompt"] = prompt
        
        st.session_state.messages.append(assistant_msg)
        log_to_db("assistant", full_response, st.session_state.active_model)
        
        st.rerun()

# --- QUIZ MODE ---
elif mode == MODE_QUIZ:
    st.header("Adaptive Calculus Quiz")
    
    if "quiz_started" not in st.session_state:
        st.session_state.quiz_started = False
        st.session_state.quiz_history = []
        st.session_state.current_question = None
        st.session_state.current_difficulty = "Easy"
        st.session_state.quiz_topic = "Derivatives"
        st.session_state.question_count = 0
        st.session_state.quiz_score = 0
        st.session_state.feedback_visible = False
        st.session_state.last_truth = ""
        st.session_state.last_solution = ""

    if not st.session_state.quiz_started:
        st.subheader("Quiz Setup")
        st.session_state.quiz_topic = st.selectbox("Select Topic:", ["Limits", "Derivatives", "Integrals"])
        st.session_state.current_difficulty = st.selectbox("Starting Difficulty:", ["Easy", "Medium", "Hard"])
        
        if st.button("Start Quiz"):
            st.session_state.quiz_started = True
            st.session_state.quiz_submitted_flag = False 
            st.rerun()

    elif st.session_state.question_count < 10:
        
        if st.session_state.current_question is None:
            with st.spinner(f"Generating Question {st.session_state.question_count + 1}/10 ({st.session_state.current_difficulty})..."):
                prompt = f"Generate a {st.session_state.current_difficulty} calculus problem about {st.session_state.quiz_topic}."
                msgs = [{"role": "system", "content": QUIZ_PROMPT}, {"role": "user", "content": prompt}]
                response = get_ai_response(msgs, st.session_state.active_model, stream=False)
                raw_q = response.choices[0].message.content
                st.session_state.current_question = sanitize_quiz_question(raw_q)
                
                log_to_db("quiz_master", st.session_state.current_question, st.session_state.active_model)
                
                st.session_state.feedback_visible = False
                st.rerun()

        progress = st.session_state.question_count / 10
        st.progress(progress, text=f"Question {st.session_state.question_count + 1} of 10")
        
        st.info(f"**Problem ({st.session_state.current_difficulty}):**")
        st.markdown(format_math(st.session_state.current_question))
        
        if not st.session_state.feedback_visible:
            with st.form(key=f"q_form_{st.session_state.question_count}"):
                user_answer = st.text_input("Your Answer:")
                submit = st.form_submit_button("Submit Answer")
                
                if submit and user_answer:
                    log_to_db("user_quiz_answer", user_answer, st.session_state.active_model)
                    
                    with st.spinner("Verifying with Python Engine..."):

                        # 1. Generate Python Code (unchanged)
                        solver_msgs = [
                            {"role": "system", "content": SOLVER_PROMPT},
                            {"role": "user", "content": f"Solve this: {st.session_state.current_question}"}
                        ]

                        code_response = get_ai_response(solver_msgs, st.session_state.active_model, stream=False)
                        generated_code = code_response.choices[0].message.content

                        # 2. Execute Python Code (unchanged)
                        truth_value = execute_math_code(generated_code)
                        st.session_state.last_truth = truth_value

                        with st.expander("🕵️ Debug: AI Internal Calculation"):
                            st.code(generated_code, language="python")
                            st.write(f"**Computed Truth:** `{truth_value}`")

                        # --- NEW: STRICT PYTHON GRADING ---
                        # We no longer ask the AI to judge. We check math directly.
                        is_correct, feedback_text = check_answer_strict(user_answer, truth_value)
                        
                        # Generate solution for explanation
                        sol_input = SOLUTION_PROMPT.format(
                            problem=st.session_state.current_question,
                            python_result=truth_value
                        )
                        sol_msgs = [
                            {"role": "system", "content": "You are a friendly calculus tutor."},
                            {"role": "user", "content": sol_input},
                        ]
                        sol_res = get_ai_response(sol_msgs, st.session_state.active_model, stream=False)
                        solution_text = sol_res.choices[0].message.content
                        st.session_state.last_solution = solution_text

                        # Update State
                        if is_correct:
                            st.session_state.quiz_score += 1
                        new_diff = adjust_difficulty(st.session_state.current_difficulty, is_correct)
                        
                        st.session_state.quiz_history.append({
                            "q": st.session_state.current_question,
                            "a": user_answer,
                            "f": feedback_text,
                            "correct": is_correct,
                            "diff": st.session_state.current_difficulty,
                            "truth": truth_value,
                            "solution": solution_text,
                        })

                        # Save individual question result to quiz_logs
                        try:
                            conn = get_db_connection()
                            cur = conn.cursor()
                            cur.execute(
                                    "INSERT INTO quiz_logs (user_id, correct, timestamp) VALUES (%s, %s, NOW());",
                                    (st.session_state.user['id'], is_correct)
                            )
                            conn.commit()
                            cur.close()
                            conn.close()
                        except Exception as e:
                            print("Failed to log quiz result:", e)

                        
                        st.session_state.current_difficulty = new_diff
                        st.session_state.feedback_visible = True
                        st.session_state.last_feedback = feedback_text
                        st.session_state.last_correct = is_correct
                        st.rerun()
                        
        else:
            st.write("---")
            if st.session_state.last_correct:
                st.success("✅ Correct! Moving difficulty up.")
            else:
                st.error("❌ Incorrect. Moving difficulty down.")
            
            st.markdown(format_math(st.session_state.last_feedback))

            # Always show solution so user can cross-check
            if st.session_state.last_truth:
                if st.session_state.last_truth.startswith("Error executing code:"):
                    st.info("Our internal math engine had trouble computing this one, but the solution below shows one correct way to solve it.")
                else:
                    st.info(f"**Correct final answer (symbolic):** `{st.session_state.last_truth}`")
            if st.session_state.last_solution:
                with st.expander("📘 View step-by-step solution"):
                    st.markdown(format_math(st.session_state.last_solution))
            
            if st.button("Next Question ➡️"):
                st.session_state.question_count += 1
                st.session_state.current_question = None
                st.rerun()

    else:
        # --- QUIZ COMPLETE ---
        
        # Only save completion ONCE using flag
        if not st.session_state.quiz_submitted_flag:
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO completed_quizzes (user_id, score, total_questions) VALUES (%s, %s, %s)",
                    (st.session_state.user['id'], st.session_state.quiz_score, 10)
                )
                conn.commit()
                cur.close()
                conn.close()
                st.session_state.quiz_submitted_flag = True  # Mark as done
            except Exception as e:
                st.error(f"Error saving final stats: {e}")

        st.balloons()
        st.success(f"🎉 Quiz Complete! Score: {st.session_state.quiz_score}/10")
        
        st.write("### Review your answers:")
        for i, item in enumerate(st.session_state.quiz_history):
            status = "✅" if item['correct'] else "❌"
            clean_title = clean_title_text(item['q'])
            with st.expander(f"{status} Q{i+1} ({item['diff']}): {clean_title}..."):
                st.write(f"**Your answer:** {item['a']}")
                st.markdown("**Grader feedback:**")
                st.markdown(format_math(item['f']))

                if item['truth'].startswith("Error executing code:"):
                    st.markdown("_Engine had trouble computing this one, but here is a worked solution:_")
                else:
                    st.markdown(f"**Correct final answer:** `{item['truth']}`")

                if item['solution']:
                    st.markdown("**Step-by-step solution:**")
                    st.markdown(format_math(item['solution']))
        
        if st.button("Start New Quiz"):
            st.session_state.quiz_started = False
            st.session_state.quiz_submitted_flag = False
            st.session_state.question_count = 0
            st.session_state.quiz_score = 0
            st.session_state.quiz_history = []
            st.rerun()

# --- GRAPH LAB MODE (Function Analyzer) ---
elif mode == MODE_GRAPH:
    st.header("📈 Function Analyzer")

    st.write("Enter a function of x. CalcFlow will plot the exact $f(x)$ you enter, plus $f'(x)$ and $\int f(x) dx$.")
    st.caption("Examples: `sec(x)`, `tanx * cosx`, `arcsin(x)`, `1/x`")

    col_main, col_side = st.columns([3, 1])

    with col_main:
        func_str = st.text_input(
            "f(x) =",
            value="",  # <--- CHANGED: Starts blank now
            placeholder="e.g. x^2 + 1", # Added placeholder text for guidance
            help="Type your function here. Supports trig (sin, cos, tan, sec, csc, cot) and inverse trig.",
        )

    with col_side:
        x_min = st.number_input("x min", value=-5.0)
        x_max = st.number_input("x max", value=5.0)

    if st.button("Analyze function"):
        if not func_str.strip():
            st.warning("Please enter a function.")
        elif x_min >= x_max:
            st.error("x min must be less than x max.")
        else:
            try:
                x = sp.symbols("x")
                cleaned = clean_function_input(func_str)
                
                # --- DEFINING THE MATH CONTEXT ---
                # This dictionary tells SymPy what "sec", "csc", etc. mean.
                safe_math = {
                    "x": x,
                    "pi": sp.pi,
                    "e": sp.E,
                    # Basic Trig
                    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                    "sec": sp.sec, "csc": sp.csc, "cot": sp.cot,
                    # Inverse Trig
                    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
                    "asec": sp.asec, "acsc": sp.acsc, "acot": sp.acot,
                    # Hyperbolic
                    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
                    "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
                    # Misc
                    "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs,
                }

                # 1. Parse RAW expression
                expr_raw = sp.sympify(cleaned, safe_math)

                # 2. Compute SIMPLIFIED version (for robust integration/analysis)
                expr_simple = sp.simplify(expr_raw)
                
                # 3. Derivatives
                f_prime_raw = sp.diff(expr_raw, x)
                f_prime_simple = sp.diff(expr_simple, x)
                f_double_simple = sp.diff(expr_simple, (x, 2))

                # 4. Integral
                f_integral = sp.integrate(expr_simple, x)

                # --- NUMERIC EVALUATION ---
                xs = np.linspace(x_min, x_max, 400)
                
                # Use RAW functions for plotting to preserve holes/asymptotes
                f_num = sp.lambdify(x, expr_raw, "numpy")
                fprime_num = sp.lambdify(x, f_prime_raw, "numpy")
                fint_num = sp.lambdify(x, f_integral, "numpy")

                ys = np.array(f_num(xs), dtype=float)
                yps = np.array(fprime_num(xs), dtype=float)
                try:
                    y_ints = np.array(fint_num(xs), dtype=float)
                except Exception:
                    y_ints = np.zeros_like(xs) 

                if not np.any(np.isfinite(ys)):
                    st.warning("Function produced no finite values on this interval.")
                    st.caption(f"Parsed as: `{cleaned}`")
                    st.stop()

                # Find critical points (approximate)
                crit_points = []
                try:
                    sols = sp.solve(sp.Eq(f_prime_simple, 0), x)
                    for s_val in sols:
                        if s_val.is_real:
                            xv = float(s_val)
                            if x_min <= xv <= x_max:
                                yv = float(expr_simple.subs(x, xv))
                                sec_deriv = float(f_double_simple.subs(x, xv))
                                kind = "min" if sec_deriv > 0 else "max" if sec_deriv < 0 else "flat"
                                crit_points.append((xv, yv, kind))
                except Exception:
                    pass

                # --- PLOTTING ---
                tab_fx, tab_fp, tab_int, tab_analysis = st.tabs(["f(x)", "f′(x)", "∫ f(x) dx", "Analysis"])

                with tab_fx:
                    fig, ax = plt.subplots()
                    # Filter out huge asymptotes for cleaner plotting
                    ys_clean = ys.copy()
                    ys_clean[np.abs(ys_clean) > 20] = np.nan
                    
                    ax.plot(xs, ys_clean, label="f(x)")
                    
                    for xv, yv, kind in crit_points:
                        color = "tab:red" if kind == "max" else "tab:green"
                        ax.scatter([xv], [yv], color=color)

                    ax.set_title(f"f(x) = {cleaned}")
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_ylim(bottom=max(np.nanmin(ys_clean) - 1, -10), top=min(np.nanmax(ys_clean) + 1, 10))
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig, use_container_width=True)

                with tab_fp:
                    fig2, ax2 = plt.subplots()
                    yps_clean = yps.copy()
                    yps_clean[np.abs(yps_clean) > 20] = np.nan
                    
                    if np.any(np.isfinite(yps_clean)):
                        ax2.plot(xs, yps_clean, label="f′(x)", color="tab:orange")
                    ax2.axhline(0, color="gray", linewidth=1)
                    ax2.set_title(f"Derivative of {cleaned}")
                    ax2.set_xlabel("x")
                    ax2.set_ylabel("f′(x)")
                    ax2.set_ylim(bottom=max(np.nanmin(yps_clean) - 1, -10), top=min(np.nanmax(yps_clean) + 1, 10))
                    ax2.grid(True)
                    st.pyplot(fig2, use_container_width=True)

                with tab_int:
                    fig3, ax3 = plt.subplots()
                    y_ints_clean = y_ints.copy()
                    y_ints_clean[np.abs(y_ints_clean) > 20] = np.nan

                    if np.any(np.isfinite(y_ints_clean)):
                        ax3.plot(xs, y_ints_clean, label="F(x)", color="tab:green")
                        ax3.set_title(f"Antiderivative ∫ {cleaned} dx")
                        ax3.set_ylim(bottom=max(np.nanmin(y_ints_clean) - 1, -10), top=min(np.nanmax(y_ints_clean) + 1, 10))
                    else:
                        st.info("Could not plot integral (values may be complex or infinite).")
                    
                    ax3.axhline(0, color="gray", linewidth=1)
                    ax3.set_xlabel("x")
                    ax3.set_ylabel("F(x)")
                    ax3.grid(True)
                    st.pyplot(fig3, use_container_width=True)

                with tab_analysis:
                    st.subheader("Input vs Simplified")
                    st.markdown(f"**Input:** `{cleaned}`")
                    st.markdown(f"**Simplified:** `{sp.simplify(expr_simple)}`")
                    
                    st.subheader("Calculus Properties")
                    st.markdown(f"**Derivative f'(x):** `{sp.simplify(f_prime_simple)}`")
                    st.markdown(f"**Integral ∫ f(x) dx:** `{sp.simplify(f_integral)} + C`")

            except Exception as e:
                st.error(f"Could not analyze that function: {e}")

# --- HANDBOOK MODE ---
elif mode == MODE_HANDBOOK:
    st.header("Calculus Formula Handbook")
    st.caption("Comprehensive reference for limits, derivatives, integrals, series, and applications.")

    tab_lim, tab_diff, tab_int, tab_trig, tab_series, tab_apps = st.tabs([
        "Limits", "Derivatives", "Integrals", "Trig", "Series", "Applications"
    ])

    # --- LIMITS TAB ---
    with tab_lim:
        st.subheader("Standard Trig Limits")
        st.markdown(r"""
        * $$ \lim_{x \to 0} \frac{\sin x}{x} = 1 $$
        * $$ \lim_{x \to 0} \frac{1 - \cos x}{x} = 0 $$
        * $$ \lim_{x \to 0} \frac{\tan x}{x} = 1 $$
        """)
        
        st.divider()

        st.subheader("General Limit Laws")
        st.markdown(r"""
        If $\lim_{x \to a} f(x) = L$ and $\lim_{x \to a} g(x) = M$, then:
        * **Sum/Diff:** $\lim_{x \to a} [f(x) \pm g(x)] = L \pm M$
        * **Product:** $\lim_{x \to a} [f(x) \cdot g(x)] = L \cdot M$
        * **Quotient:** $\lim_{x \to a} \frac{f(x)}{g(x)} = \frac{L}{M} \quad (M \neq 0)$
        * **Power:** $\lim_{x \to a} [f(x)]^n = L^n$
        """)
        
        st.info(r"**L'Hôpital's Rule:** If you get $\frac{0}{0}$ or $\frac{\infty}{\infty}$, then $\lim_{x \to a} \frac{f(x)}{g(x)} = \lim_{x \to a} \frac{f'(x)}{g'(x)}$")

        st.subheader("Theorems & Continuity")
        with st.expander("Continuity & IVT"):
            st.markdown(r"""
            **Definition of Continuity:** A function $f$ is continuous at $a$ if:
            $$ \lim_{x \to a} f(x) = f(a) $$
            
            **Intermediate Value Theorem (IVT):** If $f$ is continuous on $[a,b]$ and $N$ is between $f(a)$ and $f(b)$, there exists $c \in (a,b)$ such that $f(c) = N$.
            """)
        
        with st.expander("Squeeze Theorem"):
            st.markdown(r"""
            If $g(x) \le f(x) \le h(x)$ near $a$, and:
            $$ \lim_{x \to a} g(x) = \lim_{x \to a} h(x) = L $$
            Then $$ \lim_{x \to a} f(x) = L $$
            """)

    # --- DERIVATIVES TAB ---
    with tab_diff:
        st.subheader("Definitions & Existence Theorems")
        with st.expander("Limit Definition & MVT"):
            st.markdown(r"""
            **Limit Definition of Derivative:**
            $$ f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h} $$

            **Mean Value Theorem (MVT):**
            If $f$ is continuous on $[a,b]$ and differentiable on $(a,b)$, there exists $c \in (a,b)$ such that:
            $$ f'(c) = \frac{f(b) - f(a)}{b - a} $$

            **Rolle's Theorem:**
            If $f(a) = f(b)$, there exists $c$ where $f'(c) = 0$.
            """)

        st.divider()
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Standard Trig Derivatives")
            st.markdown(r"""
            * $\frac{d}{dx}(\sin x) = \cos x$
            * $\frac{d}{dx}(\cos x) = -\sin x$
            * $\frac{d}{dx}(\tan x) = \sec^2 x$
            * $\frac{d}{dx}(\sec x) = \sec x \tan x$
            * $\frac{d}{dx}(\csc x) = -\csc x \cot x$
            * $\frac{d}{dx}(\cot x) = -\csc^2 x$
            """)

            st.subheader("Hyperbolic Functions")
            st.markdown(r"""
            * $\frac{d}{dx}(\sinh x) = \cosh x$
            * $\frac{d}{dx}(\cosh x) = \sinh x$
            * $\frac{d}{dx}(\tanh x) = \text{sech}^2 x$
            """)

        with c2:
            st.subheader("Inverse Trig Derivatives")
            st.markdown(r"""
            * $\frac{d}{dx}(\sin^{-1} x) = \frac{1}{\sqrt{1-x^2}}$
            * $\frac{d}{dx}(\cos^{-1} x) = -\frac{1}{\sqrt{1-x^2}}$
            * $\frac{d}{dx}(\tan^{-1} x) = \frac{1}{1+x^2}$
            * $\frac{d}{dx}(\cot^{-1} x) = -\frac{1}{1+x^2}$
            * $\frac{d}{dx}(\sec^{-1} x) = \frac{1}{|x|\sqrt{x^2-1}}$
            * $\frac{d}{dx}(\csc^{-1} x) = -\frac{1}{|x|\sqrt{x^2-1}}$
            """)
            
            st.subheader("General Rules")
            st.markdown(r"""
            * **Chain Rule:** $\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)$
            * **Product Rule:** $(fg)' = f'g + fg'$
            * **Quotient Rule:** $\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}$
            """)

    # --- INTEGRALS TAB ---
    with tab_int:
        st.subheader("Fundamental Theorem of Calculus")
        st.info(r"""
        **Part 1 (Evaluation):** If $F' = f$, then $\int_a^b f(x) \, dx = F(b) - F(a)$
        
        **Part 2 (Differentiation):** $$ \frac{d}{dx} \int_a^x f(t) \, dt = f(x) $$
        
        **Average Value:** $$ f_{avg} = \frac{1}{b-a} \int_a^b f(x) \, dx $$
        """)

        st.divider()
        c_i1, c_i2 = st.columns(2)

        with c_i1:
            st.subheader("Common Trig Integrals")
            st.markdown(r"""
            * $\int \sin x \, dx = -\cos x$
            * $\int \cos x \, dx = \sin x$
            * $\int \sec^2 x \, dx = \tan x$
            * $\int \csc^2 x \, dx = -\cot x$
            * $\int \sec x \tan x \, dx = \sec x$
            * $\int \csc x \cot x \, dx = -\csc x$
            """)

            st.subheader("Advanced Trig Integrals")
            st.markdown(r"""
            * $\int \tan x \, dx = \ln|\sec x|$
            * $\int \cot x \, dx = \ln|\sin x|$
            * $\int \sec x \, dx = \ln|\sec x + \tan x|$
            * $\int \csc x \, dx = -\ln|\csc x + \cot x|$
            """)

        with c_i2:
            st.subheader("Inverse Trig Integrals")
            st.markdown(r"""
            * $\int \frac{1}{\sqrt{1-x^2}} \, dx = \sin^{-1} x$
            * $\int \frac{1}{1+x^2} \, dx = \tan^{-1} x$
            * $\int \frac{1}{|x|\sqrt{x^2-1}} \, dx = \sec^{-1} x$
            """)

            st.subheader("Basic Power/Log")
            st.markdown(r"""
            * $\int x^n \, dx = \frac{x^{n+1}}{n+1} \quad (n \neq -1)$
            * $\int \frac{1}{x} \, dx = \ln|x|$
            * $\int e^x \, dx = e^x$
            """)

    # --- TRIG CHEAT SHEET TAB ---
    with tab_trig:
        st.subheader("Useful Identities for Substitution")
        
        with st.expander("Sum & Difference Formulas"):
             st.markdown(r"""
             * $\sin(A \pm B) = \sin A \cos B \pm \cos A \sin B$
             * $\cos(A \pm B) = \cos A \cos B \mp \sin A \sin B$
             """)

        with st.expander("Pythagorean Identities", expanded=True):
            st.markdown(r"""
            * $\sin^2 x + \cos^2 x = 1$
            * $\tan^2 x + 1 = \sec^2 x$
            * $1 + \cot^2 x = \csc^2 x$
            """)

        with st.expander("Double Angle Formulas"):
            st.markdown(r"""
            * $\sin(2x) = 2\sin x \cos x$
            * $\cos(2x) = \cos^2 x - \sin^2 x$
            * $\cos(2x) = 2\cos^2 x - 1$
            * $\cos(2x) = 1 - 2\sin^2 x$
            """)
        
        with st.expander("Power Reduction (Half-Angle)"):
            st.caption("Crucial for integrating $\sin^2 x$ or $\cos^2 x$")
            st.markdown(r"""
            * $\sin^2 x = \frac{1 - \cos(2x)}{2}$
            * $\cos^2 x = \frac{1 + \cos(2x)}{2}$
            """)
            
        with st.expander("Standard Unit Circle Values"):
            st.markdown(r"""
            | $\theta$ | $0$ | $\pi/6$ | $\pi/4$ | $\pi/3$ | $\pi/2$ |
            | :--- | :---: | :---: | :---: | :---: | :---: |
            | $\sin \theta$ | $0$ | $1/2$ | $\sqrt{2}/2$ | $\sqrt{3}/2$ | $1$ |
            | $\cos \theta$ | $1$ | $\sqrt{3}/2$ | $\sqrt{2}/2$ | $1/2$ | $0$ |
            | $\tan \theta$ | $0$ | $1/\sqrt{3}$ | $1$ | $\sqrt{3}$ | $\text{und}$ |
            """)

    # --- SERIES TAB ---
    with tab_series:
        st.subheader("Common Maclaurin Series")
        st.markdown(r"""
        * $$ e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!} = 1 + x + \frac{x^2}{2!} + \dots $$
        * $$ \sin x = \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n+1}}{(2n+1)!} = x - \frac{x^3}{3!} + \dots $$
        * $$ \cos x = \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n}}{(2n)!} = 1 - \frac{x^2}{2!} + \dots $$
        * $$ \frac{1}{1-x} = \sum_{n=0}^{\infty} x^n = 1 + x + x^2 + \dots \quad (|x| < 1) $$
        """)
        
        st.subheader("Convergence Tests")
        with st.expander("Geometric & p-Series"):
            st.markdown(r"""
            **Geometric Series:** $\sum ar^n$ converges if $|r| < 1$.
            
            **p-Series:** $\sum \frac{1}{n^p}$ converges if $p > 1$.
            """)
            
        with st.expander("Ratio Test"):
            st.markdown(r"""
            Let $L = \lim_{n \to \infty} \left| \frac{a_{n+1}}{a_n} \right|$.
            * If $L < 1$: Converges absolutely.
            * If $L > 1$: Diverges.
            * If $L = 1$: Inconclusive.
            """)
            
        with st.expander("Alternating Series Test"):
            st.markdown(r"""
            $\sum (-1)^n b_n$ converges if:
            1. $b_{n+1} \le b_n$ (Decreasing)
            2. $\lim_{n \to \infty} b_n = 0$
            """)

    # --- APPLICATIONS TAB ---
    with tab_apps:
        st.subheader("Volumes of Revolution")
        c_a1, c_a2 = st.columns(2)
        
        with c_a1:
            st.markdown("**Disk Method:**")
            st.latex(r"V = \pi \int_a^b [R(x)]^2 \, dx")
            st.caption("(Rotation about x-axis, no hole)")

        with c_a2:
            st.markdown("**Washer Method:**")
            st.latex(r"V = \pi \int_a^b ([R(x)]^2 - [r(x)]^2) \, dx")
            st.caption("(Rotation with hole)")
            
        st.divider()
        st.subheader("Arc Length")
        st.latex(r"L = \int_a^b \sqrt{1 + [f'(x)]^2} \, dx")