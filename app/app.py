from flask import Flask, render_template, request

from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

app = Flask(__name__)

model_name = "t5-small"
model_ckpt = "./summarizer"
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = T5ForConditionalGeneration.from_pretrained(model_ckpt).to(device)
prefix = "summarize: "

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text-summarization', methods=["POST"])
def summarize():

    try:
        if request.method == "POST":

            inputtext = request.form["inputtext_"]

            if not inputtext.strip():
                    return render_template("output.html", data={"summary": "ERROR: No inference made. Input text can't be empty!"})

            input_text = prefix + inputtext

            tokenized_text = tokenizer.encode(input_text, return_tensors='pt', max_length=512).to(device)
            summary_ = model.generate(tokenized_text, min_length=10, max_length=40)
            summary = tokenizer.decode(summary_[0], skip_special_tokens=True)

        return render_template("output.html", data={"summary": summary})
    except Exception as e:
         return render_template("output.html", data={"summary", f"ERROR: {str(e)}"})

if __name__ == '__main__':
    app.run()