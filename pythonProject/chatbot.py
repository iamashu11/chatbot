from flask import Flask, request, jsonify, render_template
import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from keras.models import load_model

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
words = [word.lower() for word in words]
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]

    # Calculate the proportion of matched words to total words in the input sentence
    match_proportion = sum(bow) / len(sentence.split())

    ERROR_THRESHOLD = 0.75
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD or match_proportion > ERROR_THRESHOLD]

    if not results:
        # If no intents are above the threshold, return None
        return None

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# while True:
#     message = input("")
#     ints = predict_class (message)
#     if ints:
#         res = get_response(ints, intents)
#     else:
#         # If no intents are predicted above the threshold, respond with a default message
#         res = "Sorry, I have no idea what you're talking about."
#     print (res)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    message = request.form['message']
    ints = predict_class(message)
    if ints:
        res = get_response(ints, intents)
    else:
        # If no intents are predicted above the threshold, respond with a default message
        res = "Sorry, I have no idea what you're talking about."
    return jsonify({'response': res})

if __name__ == '__main__':
    app.run(debug=True)