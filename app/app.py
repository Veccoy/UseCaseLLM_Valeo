from flask import Flask, render_template, request

from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

app = Flask(__name__)

model_name = "t5-small"
model_ckpt = "./t5_summarizer"
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = T5ForConditionalGeneration.from_pretrained(model_ckpt).to(device)
prefix = "summarize: "

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text-summarization', methods=["POST"])
def summarize():

    if request.method == "POST":

        inputtext = request.form["inputtext_"]

        input_text = prefix + inputtext

        tokenized_text = tokenizer.encode(input_text, return_tensors='pt', max_length=512).to(device)
        summary_ = model.generate(tokenized_text, min_length=30, max_length=300)
        summary = tokenizer.decode(summary_[0], skip_special_tokens=True)

    return render_template("output.html", data = {"summary": summary})

if __name__ == '__main__':
    app.run()