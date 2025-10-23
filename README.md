



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
Perfect! You can include this in your **README** under a **“Demo / Example Usage”** section to show how your chatbot works with **multiple intents** and **irrelevant cases**. Here’s how you can add it:

---

### Demo / Example Usage

```text
✈ Airline Support Bot (type 'exit' to quit)

You: i want to know my flight status and also my bag is missing
Classification: Missing Bag, Flight Status
Bot (Missing Bag): I'm sorry about your missing bag. Can you provide details?
Bot (Flight Status): Let me check your flight status. Can you give me the flight number?
Was this classification correct? (yes/no): yes

You: kwndw
Classification: Irrelevant
Bot (Irrelevant): 🤔 This seems unrelated to airline queries.
Was this classification correct? (yes/no): no
Enter correct intent(s), comma separated: dsnlfkwne
❌ Intent 'dsnlfkwne' is not recognized. It must be one of the predefined intents.
⚠ No valid intents provided. Skipping update.

You: what are the discounts available now
Classification: Discounts
Bot (Discounts): Here are the available discounts: ...
Was this classification correct? (yes/no): yes

You: what are the prohibitted items and can i also take my guitar with me
Classification: Sports Music Gear, Prohibited Items Faq
Bot (Sports Music Gear): You can bring your sports/music equipment. Rules: ...
Bot (Prohibited Items Faq): Prohibited items include: ...
Was this classification correct? (yes/no): yes

You: can i take my dog, my one bag is missing and other bag is damaged
Classification: Missing Bag, Pet Travel, Damaged Bag
Bot (Missing Bag): I'm sorry about your missing bag. Can you provide details?
Bot (Pet Travel): Here are the pet travel rules: ...
Bot (Damaged Bag): I'm sorry your bag was damaged. Please provide details.
Was this classification correct? (yes/no): yes

You: i need to check the cost from Delhi to Chennai, are there any discounts, and can i take my dog and guitar with me
Classification: Discounts, Fare Check
Bot (Discounts): Here are the available discounts: ...
Bot (Fare Check): I can check fares for your flight. Which route?
Was this classification correct? (yes/no): no
Enter correct intent(s), comma separated: Discounts, Fare Check, Sports Music Gear, Pet Travel
✅ Updated intent 'Discounts' and embeddings.
✅ Updated intent 'Fare Check' and embeddings.
✅ Updated intent 'Sports Music Gear' and embeddings.
✅ Updated intent 'Pet Travel' and embeddings.
✅ Recomputed embeddings for all intents.
```

---

### ✅ Key Features Highlighted in This Demo

* **Multiple intents handled** in a single query (e.g., “Missing Bag” + “Flight Status”)
* **Irrelevant queries** classified correctly (e.g., random text → “Irrelevant”)
* **Dynamic feedback & updating**: When the user provides corrections, the **model updates embeddings**
* **Detailed responses** per intent, allowing the chatbot to answer multiple queries at once

---



