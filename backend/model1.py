from sentence_transformers import SentenceTransformer, util
import torch
from pymongo import MongoClient

# =======================================================
# 1️⃣ Initial intents
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

# =======================================================
# 2️⃣ Connect to MongoDB
# =======================================================
client = MongoClient("mongodb://localhost:27017/")
db = client["a_chatbot"]
intents_collection = db["intents"]
feedback_collection = db["feedback"]

# =======================================================
# 3️⃣ Auto-create collection & insert initial intents
# =======================================================
for intent, examples in intents_data.items():
    if intents_collection.count_documents({"intent": intent}) == 0:
        intents_collection.insert_one({"intent": intent, "examples": examples})

# =======================================================
# 4️⃣ Load Sentence Transformer Model
# =======================================================
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# =======================================================
# 5️⃣ Load all examples from DB and compute embeddings
# =======================================================
def load_examples():
    example_texts = []
    example_labels = []
    for doc in intents_collection.find():
        for text in doc["examples"]:
            example_texts.append(text)
            example_labels.append(doc["intent"])
    return example_texts, example_labels

example_texts, example_labels = load_examples()
example_embeddings = semantic_model.encode(example_texts, convert_to_tensor=True)

# =======================================================
# 6️⃣ Prediction Function
# =======================================================
def predict_airline_intent(user_text, similarity_threshold=0.55):
    user_emb = semantic_model.encode(user_text, convert_to_tensor=True)
    cosine_scores = util.cos_sim(user_emb, example_embeddings)
    max_score, idx = torch.max(cosine_scores, dim=1)
    max_score = max_score.item()
    predicted_intent = example_labels[idx.item()]
    if max_score < similarity_threshold:
        return "Irrelevant"
    return predicted_intent

# =======================================================
# 7️⃣ Update DB for new user feedback
# =======================================================
def update_intent_in_db(user_text, correct_intent):
    # Add the new example to intent examples if not exists
    intents_collection.update_one(
        {"intent": correct_intent},
        {"$addToSet": {"examples": user_text}},
        upsert=True
    )
    # Reload embeddings
    global example_texts, example_labels, example_embeddings
    example_texts, example_labels = load_examples()
    example_embeddings = semantic_model.encode(example_texts, convert_to_tensor=True)
    print(f"✅ Updated intent '{correct_intent}' in MongoDB and embeddings.")

def store_feedback(user_text, predicted_intent, correct_intent):
    feedback_entry = {
        "user_text": user_text,
        "predicted_intent": predicted_intent,
        "correct_intent": correct_intent
    }
    feedback_collection.insert_one(feedback_entry)
    print("✅ Feedback stored in MongoDB")

# =======================================================
# 8️⃣ Simple responses (can be extended)
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
# 9️⃣ Chat Function (Ask feedback every time)
# =======================================================
def chat():
    print("✈ Airline Support Bot (type 'exit' to quit)")
    while True:
        user_text = input("You: ")
        if user_text.lower() == "exit":
            break

        intent = predict_airline_intent(user_text)
        print(f"Bot: {get_response(intent)}")
        print(f"(Predicted Intent: {intent})")

        # Always ask for feedback
        feedback = input("Was that correct? (yes/no): ").lower()
        if feedback == "no":
            correct_intent = input("Enter correct intent: ").strip()
        else:
            correct_intent = intent

        # Store feedback and update DB
        store_feedback(user_text, intent, correct_intent)
        update_intent_in_db(user_text, correct_intent)

# =======================================================
# 10️⃣ Run chatbot
# =======================================================
chat()
