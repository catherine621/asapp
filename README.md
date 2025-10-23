



# Airline Chatbot - Intent Classification

## Project Overview

This project leverages **Sentence Transformers** for semantic understanding. The system allows a chatbot to:

* Understand the **meaning of sentences**, not just keywords
* Perform **intent classification**
* Enable **semantic search** for similar queries

The project uses a **Streamlit frontend**, **FastAPI backend**, and **MongoDB** for storage.

---

## Tech Stack

* **Python 3.10+**
* **Streamlit** – Frontend UI
* **FastAPI** – Backend API
* **MongoDB Atlas** – Database
* **Sentence Transformers (SBERT)** – ML Model

---

## Folder Structure

```
asapp/
├── backend/       # FastAPI backend code
├── frontend/      # Streamlit frontend code
├── .gitignore
```

---

## Installation

1. Clone the repo:

```bash
git clone https://github.com/catherine621/asapp.git
cd asapp
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Setup

1. **MongoDB Atlas**:

   * Create a free cluster
   * Add your cluster URI in `backend/config.py` or `.env`

2. **Frontend**:

   * Ensure API endpoints in the frontend point to your backend

3. **ML Model**:

   * Preload Sentence Transformer model in backend code (`ml_model.py`)

---

## Running the Project

### Start Backend

```bash
cd backend
uvicorn main:app --reload
```

### Start Frontend

```bash
cd frontend
streamlit run app.py
```

### Access App

* Open browser at `http://localhost:8501`

---

## Notes

* Ensure **MongoDB URI** is correct
* Python 3.10+ recommended
* For large datasets, consider caching embeddings

---


