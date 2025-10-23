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
db = client["cathychatbot"]
intents_collection = db["intents"]
feedback_collection = db["feedback"]

# =======================================================
# 3️⃣ Insert initial intents if not present
# =======================================================
for intent, examples in intents_data.items():
    if intents_collection.count_documents({"intent": intent}) == 0:
        intents_collection.insert_one({"intent": intent, "examples": examples})

# =======================================================
# 4️⃣ Load Sentence Transformer
# =======================================================
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# =======================================================
# 5️⃣ Load examples and compute embeddings
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

# Precompute intent-wise embeddings for multi-intent prediction
def load_intent_embeddings():
    embeddings = {}
    for doc in intents_collection.find():
        examples = doc["examples"]
        if examples:
            embeddings[doc["intent"]] = semantic_model.encode(examples, convert_to_tensor=True)
    return embeddings

intent_embeddings = load_intent_embeddings()

# =======================================================
# 6️⃣ Multi-intent prediction
# =======================================================

# =======================================================
# Recompute embeddings for all intents
# =======================================================
def update_embeddings_for_all_intents():
    """
    Recompute embeddings for all intents after feedback update
    """
    global intent_embeddings
    intent_embeddings = {}
    for doc in intents_collection.find():
        examples = doc["examples"]
        if examples:
            intent_embeddings[doc["intent"]] = semantic_model.encode(examples, convert_to_tensor=True)
    print("✅ Recomputed embeddings for all intents.")



def predict_multiple_intents(user_text, similarity_threshold=0.6, top_k=3):
    """
    Returns multiple relevant intents based on max similarity per intent.
    Automatically filters out weakly related intents.
    """
    global intent_embeddings  # Make sure we use the updated embeddings
    user_emb = semantic_model.encode(user_text, convert_to_tensor=True)
    intent_scores = {}

    # Compute max similarity for each intent
    for intent, example_embs in intent_embeddings.items():
        scores = util.cos_sim(user_emb, example_embs)
        intent_scores[intent] = torch.max(scores).item()

    # ✅ Filter out low-confidence intents
    relevant_intents = {
        intent: score for intent, score in intent_scores.items() if score >= similarity_threshold
    }

    # ✅ Sort by score descending
    sorted_intents = sorted(relevant_intents.items(), key=lambda x: x[1], reverse=True)

    # ✅ Pick top_k results
    predicted_intents = [intent for intent, _ in sorted_intents[:top_k]]

    # ✅ Handle case where only one strong match exists
    if len(predicted_intents) == 0:
        predicted_intents = ["Irrelevant"]

    return predicted_intents

# =======================================================
# 7️⃣ Update DB & embeddings after feedback
# =======================================================
def update_intent(user_text, correct_intent):
    intents_collection.update_one(
        {"intent": correct_intent},
        {"$addToSet": {"examples": user_text}},
        upsert=True
    )
    # Update all embeddings immediately
    global example_texts, example_labels, example_embeddings, intent_embeddings
    example_texts, example_labels = load_examples()
    example_embeddings = semantic_model.encode(example_texts, convert_to_tensor=True)
    intent_embeddings = load_intent_embeddings()
    print(f"✅ Updated intent '{correct_intent}' and embeddings.")

def store_feedback(user_text, predicted_intent, correct_intent):
    feedback_collection.insert_one({
        "user_text": user_text,
        "predicted_intent": predicted_intent,
        "correct_intent": correct_intent
    })

# =======================================================
# 8️⃣ Responses
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
# 9️⃣ Chat function
# =======================================================
def chat():
    print("✈ Airline Support Bot (type 'exit' to quit)")
    while True:
        user_text = input("You: ")
        if user_text.lower() == "exit":
            break

        predicted_intents = predict_multiple_intents(user_text)
        print(f"Classification: {', '.join(predicted_intents)}")

        for intent in predicted_intents:
            print(f"Bot ({intent}): {get_response(intent)}")

        feedback = input("Was this classification correct? (yes/no): ").lower()
        if feedback == "no":
            correct_input = input("Enter correct intent(s), comma separated: ").strip()
            correct_intents = [ci.strip() for ci in correct_input.split(",")]
            # List of all allowed intents
            allowed_intents = list(intents_data.keys())

            # Validate each intent
            valid_intents = []
            for ci in correct_intents:
                if ci not in allowed_intents:
                    print(f"❌ Intent '{ci}' is not recognized. It must be one of the predefined intents.")
                else:
                    valid_intents.append(ci)

            # If none of the intents are valid, skip update
            if not valid_intents:
                print("⚠ No valid intents provided. Skipping update.")
            else:
                # Update DB for valid intents
                for correct in valid_intents:
                    update_intent(user_text, correct)

                # Recompute embeddings
                update_embeddings_for_all_intents()

                # Store feedback
                for predicted, correct in zip(predicted_intents, valid_intents):
                    store_feedback(user_text, predicted, correct)

# =======================================================
# 10️⃣ Run chatbot
# =======================================================
if __name__ == "__main__":
    chat()
