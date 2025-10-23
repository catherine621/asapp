import streamlit as st
st.set_page_config(page_title="Airline Chatbot", page_icon="✈️", layout="centered")

from sentence_transformers import SentenceTransformer, util
import torch
from pymongo import MongoClient

# =======================================================
# 1️⃣ MongoDB Connection
# =======================================================
client = MongoClient("mongodb://localhost:27017/")
db = client["a_chatbot"]
intents_collection = db["intents"]
feedback_collection = db["feedback"]

# =======================================================
# 2️⃣ Default Intent Examples (Bootstrap DB)
# =======================================================
intents_data = {
    "Cancel Trip": ["I want to cancel my flight", "Please cancel my booking", "Cancel my ticket"],
    "Cancellation Policy": ["What is your cancellation policy?", "Tell me about ticket cancellation rules"],
    "Carry On Luggage Faq": ["What items can I carry in hand luggage?", "Is laptop allowed in cabin bag?"],
    "Change Flight": ["I need to change my flight", "Can I reschedule my flight?"],
    "Check In Luggage Faq": ["What is the check-in baggage limit?", "How many bags can I check in?"],
    "Complaints": ["I want to report an issue", "The staff was rude"],
    "Damaged Bag": ["My bag was damaged", "The luggage got broken"],
    "Discounts": ["Are there any discounts?", "Tell me about special offers"],
    "Fare Check": ["What is the fare for Delhi to Mumbai?", "Show me ticket prices"],
    "Flight Status": ["Is my flight on time?", "Has my flight been delayed?"],
    "Flights Info": ["Show me flights to London", "Which flights are available to Paris?"],
    "Insurance": ["Do you provide travel insurance?", "Tell me about flight insurance"],
    "Medical Policy": ["What is your medical policy?", "Can I travel if I am sick?"],
    "Missing Bag": ["I lost my luggage", "My bag didn’t arrive"],
    "Pet Travel": ["Can I travel with my pet?", "Tell me about pet travel rules"],
    "Prohibited Items Faq": ["What items are prohibited?", "Can I carry knives or scissors?"],
    "Seat Availability": ["Are seats available for tomorrow?", "Check seat availability for flight AI101"],
    "Sports Music Gear": ["Can I bring my guitar on the plane?", "What about sports equipment?"],
    "Irrelevant": ["Find the square root of 64", "Tell me about Tesla cars"]
}

for intent, examples in intents_data.items():
    if intents_collection.count_documents({"intent": intent}) == 0:
        intents_collection.insert_one({"intent": intent, "examples": examples})

# =======================================================
# 3️⃣ Load Model
# =======================================================
model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

# =======================================================
# 4️⃣ Predict Intent Function
# =======================================================
def predict_intents(user_text, threshold=0.6, top_k=2):
    user_emb = model.encode(user_text, convert_to_tensor=True)
    intent_scores = {}

    for doc in intents_collection.find():
        intent = doc["intent"]
        examples = doc["examples"]
        if not examples:
            continue
        ex_emb = model.encode(examples, convert_to_tensor=True)
        score = torch.max(util.cos_sim(user_emb, ex_emb)).item()
        intent_scores[intent] = score

    relevant = {i: s for i, s in intent_scores.items() if s >= threshold}
    sorted_intents = sorted(relevant.items(), key=lambda x: x[1], reverse=True)
    return [i for i, _ in sorted_intents[:top_k]] if sorted_intents else ["Irrelevant"]

# =======================================================
# 5️⃣ Responses
# =======================================================
responses = {
    "Cancel Trip": "I can help you cancel your flight. Please provide your booking details.",
    "Cancellation Policy": "Here is our cancellation policy: ...",
    "Carry On Luggage Faq": "You can carry these items in your hand luggage: ...",
    "Change Flight": "Sure, we can reschedule your flight. What date would you like?",
    "Check In Luggage Faq": "Check-in baggage rules: ...",
    "Complaints": "Please tell me your complaint. We'll resolve it ASAP.",
    "Damaged Bag": "I'm sorry your bag was damaged. Please provide details.",
    "Discounts": "Here are the available discounts: ...",
    "Fare Check": "I can check fares for your flight. Which route?",
    "Flight Status": "Let me check your flight status. Can you give me the flight number?",
    "Flights Info": "Here is the flight info you requested: ...",
    "Insurance": "We provide travel insurance. Details: ...",
    "Medical Policy": "Our medical policy is: ...",
    "Missing Bag": "I'm sorry about your missing bag. Can you provide details?",
    "Pet Travel": "Here are the pet travel rules: ...",
    "Prohibited Items Faq": "Prohibited items include: ...",
    "Seat Availability": "I can check seat availability. Which flight?",
    "Sports Music Gear": "You can bring your sports/music equipment. Rules: ...",
    "Irrelevant": "🤔 This seems unrelated to airline queries."
}

# =======================================================
# 6️⃣ DB Feedback Functions
# =======================================================
def store_feedback(user_text, predicted, correct):
    feedback_collection.insert_one({
        "user_text": user_text,
        "predicted": predicted,
        "correct": correct
    })

def update_intent_db(user_text, intent):
    intents_collection.update_one(
        {"intent": intent},
        {"$addToSet": {"examples": user_text}},
        upsert=True
    )

# =======================================================
# 7️⃣ Streamlit UI
# =======================================================
st.title("✈️ Airline Support Chatbot")
st.markdown("Ask about your flight, baggage, cancellation, or travel info below 👇")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("💬 Type your message:")

if user_input:
    predicted = predict_intents(user_input)
    st.session_state.chat_history.append({
        "user": user_input,
        "predicted": predicted
    })

# --- Show conversation ---
for chat in st.session_state.chat_history:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Classification:** {', '.join(chat['predicted'])}")

    for intent in chat["predicted"]:
        st.markdown(f"**Bot ({intent}):** {responses[intent]}")

    # --- Feedback section ---
    st.markdown(f"Was this classification correct?")
    fb_key = f"fb_{chat['user']}"
    feedback = st.radio("", ["Yes", "No"], key=fb_key, horizontal=True, label_visibility="collapsed")

    if feedback == "No":
        correct_intent = st.selectbox(
            f"Select correct intent for: '{chat['user']}'",
            list(intents_data.keys()),
            key=f"sel_{chat['user']}"
        )
    else:
        correct_intent = ", ".join(chat["predicted"])

    if st.button(f"Submit Feedback for '{chat['user']}'", key=f"btn_{chat['user']}"):
        store_feedback(chat["user"], chat["predicted"], correct_intent)
        update_intent_db(chat["user"], correct_intent)
        st.success("✅ Feedback saved successfully!")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Sentence Transformers, and MongoDB.")
