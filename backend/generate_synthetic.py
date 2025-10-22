# generate_synthetic_all_intents.py
# Generates synthetic examples for all airline intents and inserts into MongoDB

from pymongo import MongoClient
import os
from dotenv import load_dotenv
import random

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
DB_NAME = os.getenv('DB_NAME', 'air_bot')

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
DATA_COLL = db['data']

# Define all 18 intents with a few seed examples each
intents = {
    'Cancel Trip': ["I want to cancel my flight", "Please cancel my booking", "I need to cancel my reservation"],
    'Cancellation Policy': ["What is your cancellation policy?", "Tell me about ticket cancellation rules"],
    'Carry On Luggage Faq': ["What items can I carry in hand luggage?", "Is laptop allowed in cabin bag?"],
    'Change Flight': ["I need to change my flight", "Can I reschedule my flight?"],
    'Check In Luggage Faq': ["What is the check-in baggage limit?", "How many bags can I check in?"],
    'Complaints': ["I want to report an issue", "The staff was rude"],
    'Damaged Bag': ["My bag was damaged", "The luggage got broken"],
    'Discounts': ["Are there any discounts?", "Tell me about special offers"],
    'Fare Check': ["What is the fare for Delhi to Mumbai?", "Show me ticket prices"],
    'Flight Status': ["Is my flight on time?", "Has my flight been delayed?"],
    'Flights Info': ["Show me flights to London", "Which flights are available to Paris?"],
    'Insurance': ["Do you provide travel insurance?", "Tell me about flight insurance"],
    'Medical Policy': ["What is your medical policy?", "Can I travel if I am sick?"],
    'Missing Bag': ["I lost my luggage", "My bag didn’t arrive"],
    'Pet Travel': ["Can I travel with my pet?", "Tell me about pet travel rules"],
    'Prohibited Items Faq': ["What items are prohibited?", "Can I carry knives or scissors?"],
    'Seat Availability': ["Are seats available for tomorrow?", "Check seat availability for flight AI101"],
    'Sports Music Gear': ["Can I bring my guitar on the plane?", "What about sports equipment?"]
}

if __name__ == '__main__':
    entries = []

    for label, examples in intents.items():
        for ex in examples:
            # Generate 50+ synthetic variants per example
            for i in range(50):
                text = ex
                if random.random() < 0.4:
                    text += ' ' + random.choice(['please', 'ASAP', 'now'])
                if random.random() < 0.2:
                    text = 'Hi, ' + text
                
                # Avoid duplicates
                if DATA_COLL.find_one({'text': text, 'label': label}):
                    continue
                
                entries.append({
                    'text': text,
                    'label': label,
                    'synthetic': True
                })

    if entries:
        DATA_COLL.insert_many(entries)
        print(f'✅ Inserted {len(entries)} synthetic examples for all intents.')
    else:
        print("No new entries were generated (all already exist).")
