import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
from pymongo import MongoClient

# =======================================================
# 1️⃣ Connect to MongoDB
# =======================================================
client = MongoClient("mongodb://localhost:27017/")
db = client["a_chatbot"]
intents_collection = db["intents"]
feedback_collection = db["feedback"]

# =======================================================
# 2️⃣ Initialize intents if DB empty
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
# 3️⃣ Load Sentence Transformer model
# =======================================================
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# =======================================================
# 4️⃣ Load examples and compute embeddings
# =======================================================
def load_examples():
    texts, labels = [], []
    for doc in intents_collection.find():
        for text in doc["examples"]:
            texts.append(text)
            labels.append(doc["intent"])
    return texts, labels

example_texts, example_labels = load_examples()
example_embeddings = semantic_model.encode(example_texts, convert_to_tensor=True)

# =======================================================
# 5️⃣ Prediction function
# =======================================================
def predict_intent(user_text, threshold=0.55):
    user_emb = semantic_model.encode(user_text, convert_to_tensor=True)
    scores = util.cos_sim(user_emb, example_embeddings)
    max_score, idx = torch.max(scores, dim=1)
    max_score = max_score.item()
    predicted = example_labels[idx.item()]
    if max_score < threshold:
        predicted = "Irrelevant"
    return predicted

# =======================================================
# 6️⃣ Update MongoDB and embeddings
# =======================================================
def update_intent_db(user_text, correct_intent):
    intents_collection.update_one(
        {"intent": correct_intent},
        {"$addToSet": {"examples": user_text}},
        upsert=True
    )
    global example_texts, example_labels, example_embeddings
    example_texts, example_labels = load_examples()
    example_embeddings = semantic_model.encode(example_texts, convert_to_tensor=True)

def store_feedback(user_text, predicted_intent, correct_intent):
    feedback_collection.insert_one({
        "user_text": user_text,
        "predicted_intent": predicted_intent,
        "correct_intent": correct_intent
    })

# =======================================================
# 7️⃣ Responses
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

def get_response(intent):
    return responses.get(intent, "Hmm... I didn’t understand that.")

# =======================================================
# 8️⃣ Streamlit UI
# =======================================================
st.set_page_config(page_title="Airline Chatbot", page_icon="✈️")
st.title("✈ Airline Support Bot")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("You:", "")

if user_input:
    predicted_intent = predict_intent(user_input)
    response = get_response(predicted_intent)

    # Show bot response
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", f"{response} (Intent: {predicted_intent})"))

    # Ask for feedback using radio buttons
    feedback = st.radio("Was this correct?", ("Yes", "No"), key=user_input)

    if feedback == "No":
        correct_intent = st.selectbox("Select correct intent:", list(intents_data.keys()), key="correct_"+user_input)
    else:
        correct_intent = predicted_intent

    # Save feedback permanently
    if st.button("Submit Feedback", key="submit_"+user_input):
        store_feedback(user_input, predicted_intent, correct_intent)
        update_intent_db(user_input, correct_intent)
        st.success("✅ Feedback stored and model updated!")

# Display chat
for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**{sender}:** {message}")
    else:
        st.markdown(f"**{sender}:** {message}")
