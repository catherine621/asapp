from flask import Flask, request, jsonify
from pymongo import MongoClient
import datetime
import os

app = Flask(__name__)

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client['airline_bot']
FEEDBACK_COLL = db['feedback']
DATA_COLL = db['data']
METRICS_COLL = db['metrics']

# Load model
from model import IntentModel
model = IntentModel()
model.load()  # load model if exists

@app.route('/')
def index():
    return "Flask server is running. Use /api/... endpoints."


# --- CLASSIFY ---
@app.route('/api/classify', methods=['POST'])
def classify():
    body = request.json or {}
    text = body.get('text', '').strip()
    if not text:
        return jsonify({'error': 'text required'}), 400

    pred = model.predict([text])
    confidence = 1.0  # optional
    result = {'text': text, 'intent': pred[0], 'confidence': confidence, 'ts': datetime.datetime.utcnow()}
    
    # Save classified data to database
    DATA_COLL.insert_one({
        'text': text,
        'label': pred[0],
        'synthetic': False,
        'ts': datetime.datetime.utcnow()
    })

    return jsonify(result)


# --- FEEDBACK ---
@app.route('/api/feedback', methods=['POST'])
def feedback():
    body = request.json or {}
    text = body.get('text', '').strip()
    pred = body.get('pred')
    correct = body.get('correct')
    true_label = body.get('true_label')

    if not text or pred is None or correct is None:
        return jsonify({'error': 'text, pred and correct required'}), 400

    doc = {
        'text': text,
        'pred': pred,
        'correct': bool(correct),
        'true_label': true_label,
        'ts': datetime.datetime.utcnow()
    }
    FEEDBACK_COLL.insert_one(doc)

    if not correct and true_label:
        DATA_COLL.insert_one({
            'text': text,
            'label': true_label,
            'synthetic': False,
            'ts': datetime.datetime.utcnow()
        })

    return jsonify({'status': 'ok'})


# --- TRAIN MODEL ---
@app.route('/api/train', methods=['POST'])
def train():
    docs = list(DATA_COLL.find())
    if len(docs) < 10:
        return jsonify({'error': 'need at least 10 training examples in data collection'}), 400

    texts = [d['text'] for d in docs]
    labels = [d['label'] for d in docs]

    acc, report = model.train(texts, labels)
    model.save()

    METRICS_COLL.insert_one({
        'ts': datetime.datetime.utcnow(),
        'type': 'train',
        'accuracy': acc,
        'report': report
    })

    return jsonify({'status': 'trained', 'accuracy': acc, 'report': report})


# --- METRICS ---
@app.route('/api/metrics', methods=['GET'])
def metrics():
    q = list(METRICS_COLL.find().sort('ts', -1).limit(200))
    for d in q:
        d['_id'] = str(d['_id'])
        if 'ts' in d:
            d['ts'] = d['ts'].isoformat()
    return jsonify({'metrics': q})


# --- NEW: GET CLASSIFIED DATA ---
@app.route('/api/classified', methods=['GET'])
def get_classified_data():
    """Return all classified data from MongoDB"""
    data = list(DATA_COLL.find().sort('ts', -1))
    for d in data:
        d['_id'] = str(d['_id'])
        if 'ts' in d:
            d['ts'] = d['ts'].isoformat()
    return jsonify({'classified_data': data})

@app.route('/api/synthetic/correct', methods=['GET'])
def get_correct_synthetic():
    """Fetch only correctly classified synthetic samples"""
    data = list(DATA_COLL.find({'synthetic': True}))
    
    correct = []
    for d in data:
        text = d['text']
        true_label = d['label']
        pred = model.predict([text])[0]

        if pred == true_label:  # correct classification
            d['_id'] = str(d['_id'])
            if 'ts' in d:
                d['ts'] = d['ts'].isoformat()
            d['pred'] = pred
            correct.append(d)

    return jsonify({'correct_synthetic_data': correct})



if __name__ == '__main__':
    app.run(host=os.getenv('HOST', '0.0.0.0'), port=int(os.getenv('PORT', 5000)))
