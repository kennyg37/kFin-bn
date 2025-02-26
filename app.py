from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from flask_cors import CORS


load_dotenv()
port = os.getenv("PORT")

app = Flask(__name__)

@app.route('/', methods=['GET'])
def app_initiation():
    return 'Welcome to KinFin'

@app.route('/translate', methods=['POST'])
def translate():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = "mbazaNLP/Nllb_finetuned_education_en_kin"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    data = request.get_json()
    text = data['text']

    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({'translation': output_text})

if __name__ == '__main__':
    app.run( host='0.0.0.0', port=port)