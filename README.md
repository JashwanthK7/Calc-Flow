# CalcFlow: AI Calculus Tutor

CalcFlow is an **AI-powered interactive tutor for Calculus I**. It uses **local LLMs via Ollama** to deliver Socratic-style tutoring, adaptive quizzes, and step-by-step problem solving with **symbolic Python verification**.

---

## Features

* **AI Chat Tutor**
  Ask calculus questions and receive clear, LaTeX-formatted explanations.

* **Adaptive Quiz Mode**
  Questions dynamically adjust in difficulty based on your performance.

* **Auto-Grading**
  Uses Python (`sympy`) to *mathematically verify* answers rather than relying on text matching.

* **Local Privacy**
  Runs entirely on your machine using Ollama — no external API calls.

---

## Prerequisites

Ensure the following are installed before running the app:

* **Python 3.10+**

* **Ollama** (installed and running)

* **Local LLM Model**

  ```bash
  ollama pull llama3
  ```

  *(You may also use `mistral`)*

* **PostgreSQL (Optional but Recommended)**
  Used for storing chat logs and quiz history.

  **Default configuration:**

  * Database name: `tutor_app`
  * User: `postgres`
  * Password: `password`
    *(You can modify these in `setup_env_local.ps1` or the shell script)*

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Calc-flow.git
cd Calc-flow
```

---

### 2️. Set Up Python Virtual Environment

**Windows**

```bash
python -m venv venv
.\venv\Scripts\activate
```

**Mac / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3️. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the App

### 1️. Configure Environment Variables

* **Windows**

  ```bash
  .\setup_env_local.ps1
  ```

* **Mac / Linux**

  ```bash
  source setup_env_local.sh
  ```

  *(Create this file if it does not exist)*

---

### 2️. Launch the Application

```bash
streamlit run streamlit_app.py --server.port 8501
```

---

### 3️. Open in Browser

Navigate to:

```
http://localhost:8501
```

---

## Docker (PostgreSQL Only)

If you prefer **not** to install PostgreSQL locally, you can run it using Docker:

```bash
docker run \
  --name local-postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=tutor_app \
  -p 5432:5432 \
  -d postgres
```

> Make sure Docker is running before executing the above command.

---

## Notes

* CalcFlow is designed for **Calculus I** topics.
* All LLM inference happens **locally** via Ollama.
* PostgreSQL is optional but recommended for persistence.

---

Happy Learning with **CalcFlow** 
