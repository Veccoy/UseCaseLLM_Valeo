"""Demo application for LLM summarization task."""

from flask import Flask, render_template, request

from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

app = Flask(__name__)

MODEL_NAME = "t5-small"
MODEL_CKPT = "./summarizer"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = T5ForConditionalGeneration.from_pretrained(MODEL_CKPT).to(DEVICE)
PREFIX = "summarize: "


@app.route('/')
def home():
    """Home page to enter the text to summarize"""
    return render_template('index.html')


@app.route('/text-summarization', methods=["POST"])
def summarize():
    """Sumarization page to see the prediction of the model"""
    try:
        if request.method == "POST":

            inputtext = request.form["inputtext_"]

            if not inputtext.strip():
                return render_template("output.html",
                                       data={"summary":
                                             "ERROR: No inference made."
                                             "Input text can't be empty!"})

            input_text = PREFIX + inputtext

            tokenized_text = TOKENIZER.encode(input_text,
                                              return_tensors='pt',
                                              max_length=512
                                              ).to(DEVICE)
            summary_ = MODEL.generate(tokenized_text,
                                      min_length=10,
                                      max_length=40)
            summary = TOKENIZER.decode(summary_[0], skip_special_tokens=True)

            return render_template("output.html", data={"summary": summary})

        return render_template("output.html",
                               data={"summary",
                                     "ERROR: Other Request method"
                                     "than POST received."})
    except Exception as e:  # noqa: W0718
        return render_template("output.html",
                               data={"summary", f"ERROR: {str(e)}"})


if __name__ == '__main__':
    app.run()
