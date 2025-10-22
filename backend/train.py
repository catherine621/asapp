# train.py - CLI helper to trigger training locally (train and save model)

from pymongo import MongoClient
from dotenv import load_dotenv
import os
from model import IntentModel

# Load environment variables
load_dotenv()

# MongoDB connection setup
MONGO_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
DB_NAME = os.getenv('DB_NAME', 'airline_bot')

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
DATA_COLL = db['data']

if __name__ == '__main__':
    # Fetch training data from MongoDB
    docs = list(DATA_COLL.find())

    if len(docs) < 10:
        print("❌ Not enough training data. At least 10 examples required.")
        exit(1)

    texts = [d['text'] for d in docs]
    labels = [d['label'] for d in docs]

    # Initialize and train the model
    model = IntentModel()
    acc, report = model.train(texts, labels)
    model.save()

    # Display training results
    print("✅ Model trained successfully!")
    print("Accuracy:", acc)
    print("Report:", report)
