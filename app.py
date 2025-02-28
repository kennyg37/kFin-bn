from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer
import torch

load_dotenv()
port = int(os.getenv("PORT", 8080))

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ktokenizer, kmodel, ftokenizer, fmodel
    
    # Kinyarwanda model
    kmodel_name = "mbazaNLP/Nllb_finetuned_education_en_kin"
    ktokenizer = AutoTokenizer.from_pretrained(kmodel_name)
    kmodel = AutoModelForSeq2SeqLM.from_pretrained(kmodel_name)

    # finQA model
    fmodel_path = "kennyg37/small_t5_finetuned_finqa"
    fmodel = AutoModelForSeq2SeqLM.from_pretrained(fmodel_path)
    ftokenizer = T5Tokenizer.from_pretrained("t5-small")

    print("Models loaded successfully!")
    yield  

    del kmodel, fmodel, ktokenizer, ftokenizer
    print("Models unloaded!")

app = FastAPI(lifespan=lifespan)

class TextInput(BaseModel):
    text: str

@app.get("/")
def app_initiation():
    return {"message": "Welcome to KFin"}

@app.post("/translate")
def translate(data: TextInput):
    try:
        inputs = ktokenizer(data.text, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        outputs = kmodel.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
        output_text = ktokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {"translation": output_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
def generate_text(data: TextInput):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fmodel.to(device)
        
        input_text = f"question: {data.text}"
        inputs = ftokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()} 

        outputs = fmodel.generate(
            **inputs,
            max_length=128,
            do_sample=True, 
            temperature=0.7, 
            top_k=50,  
            top_p=0.9,  
            repetition_penalty=1.2,  
            no_repeat_ngram_size=4,  
        )
        response = ftokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/generate/kinyarwanda")
def translate_generated(data: TextInput):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fmodel.to(device)
        
        input_text = f"question: {data.text}"
        inputs = ftokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()} 

        outputs = fmodel.generate(
            **inputs,
            max_length=128,
            do_sample=True, 
            temperature=0.7, 
            top_k=50,  
            top_p=0.9,  
            repetition_penalty=1.2,  
            no_repeat_ngram_size=4,  
        )
        response = ftokenizer.decode(outputs[0], skip_special_tokens=True)
        
        inputs = ktokenizer(response, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        outputs = kmodel.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
        output_text = ktokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {"Kinyarwanda output": output_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
