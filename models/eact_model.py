from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def eact_translate(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(
        **inputs,
        num_beams=6,          # more robust decoding
        length_penalty=1.2,
        early_stopping=True
    )
    return tokenizer.decode(translated[0], skip_special_tokens=True)
