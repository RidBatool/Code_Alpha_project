
from flask import Flask, request, jsonify, render_template
import random
import string
import spacy
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Load GPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Sample responses
responses = {
    "greet": ["Hello!", "Hi there!", "Greetings!", "How can I assist you?"],
    "bye": ["Goodbye!", "See you later!", "Take care!"],
    "default": ["I'm not sure how to respond to that.", "Can you rephrase?", "Let's talk about something else."]
}

chat_history_ids = None  # Initialize chat history

def respond_with_nlp(message, chat_history_ids):
    doc = nlp(message.lower().translate(str.maketrans('', '', string.punctuation)))

    # Check for greetings
    if any(token.lemma_ in ["hello", "hi"] for token in doc):
        return random.choice(responses["greet"]), chat_history_ids
    # Check for farewells
    elif any(token.lemma_ == "bye" for token in doc):
        return random.choice(responses["bye"]), chat_history_ids
    else:
        return get_gpt_response(message, chat_history_ids)

def get_gpt_response(message, chat_history_ids):
    new_user_input_ids = tokenizer.encode(message + tokenizer.eos_token, return_tensors='pt')

    if chat_history_ids is None:
        bot_input_ids = new_user_input_ids
    else:
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

    attention_mask = torch.ones(bot_input_ids.shape, device=bot_input_ids.device)

    chat_history_ids = model.generate(
        bot_input_ids,
        attention_mask=attention_mask,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global chat_history_ids
    user_message = request.json['message']

    # Process message and get response
    response, chat_history_ids = respond_with_nlp(user_message, chat_history_ids)

    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=False)
