# Practical Use Case : Fine-tune a LLM

## Project Description

The objective is to create a comprehensive GitHub repository that showcases my expertise in leveraging Generative AI models, with a particular focus on deploying, optimizing, and fine-tuning Large Language Models (LLMs).
The project involves selecting a pre-trained T5 model from Hugging Face, applying advanced optimization techniques, retraining the model on teh XSum dataset, evaluating its performance, and seamlessly integrating the workflow into a CI/CD pipeline. All the choices made have been detailled and explained below.

## Setup Instructions

To setup the environment to use the fine-tuned T5-small model, please use the Dockerfile provided and follow these steps after cloning the repository:

1 - Build the Docker image:
```bash
cd <path_to_repository>
docker build -t <image_name> -f ./docker/Dockerfile .
```

2 - Run a container from the created image with 1 GPU :
```bash
docker run -it --gpus 1 --shm-size=4gb --name test llm_test:latest
```

3 - To run the application inside the container use this command line as the `app.py` file should be located in the home folder:
```bash
python app.py
```

If you do not use Docker, you can find a Python requirement file with all the libraries you need to run the project codes. Then, use this command line:
```bash
pip install -r requirements.txt
```

## User Guide

Once you launch the application, using the third command line in the previous part, you can access the home page following this link by default: http://127.0.0.1:5000/.

Then, you can submit some text to be summarized by clicking on the textbox and writing the text to be summarized. Then, click on the "Summarize Text" and you will be redirected to another page containing the summarized text. This text will be summarized using the fine-tuned T5-small model obtained in the (training notebook)[./training/training.ipynb].

Once you read the summary, you can return to the home page using the "Go Back" button to submit another text to be summarized.

You can use this application to summarize articles, documents or any other large amount of text. Everything (Hugging Face model handling, input/output, CSS design) is managed by the Flask application.

### 1. Model Selection

I have selected a pre-trained T-small model from the Hugging Face Model Hub. The T5 model was presented in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf) by Colin Raffelet al. and is an encoder-decoder model pre-trained on a multi-task mixture of unsupervised and supervised tasks and for which each task is converted into a text-to-text format.

T5 works well on a variety of tasks out-of-the-box. A different prefix has to be prepended to the input for each task, e.g., for translation or summarization. Thus, it is very well suited for summarization tasks for which several datasets exist in the Hugging Face Hub. In our case, the prefix for summarization task will be `summarize: `.

Moreover, it comes in different sizes: small, base, large, 3b, 11b, which makes it very scalable. It will be very useful for this practical use case as I have used the free GPUs from Google Colab (T4).

As the model is pre-trained on a mixture of unsupervised and supervised tasks, it has the potential to generalize well to new tasks. Furthermore, the ability of T5 to perform multiple NLP tasks by simply changing the prefix of the input is a significant advantage. Its performance on various tasks has made it one of the most promising approaches for NLP applications, with impressive results in text summarization indeed.

### 2. Model Fine-Tuning

I have fine-tuned the selected T5-small model on a text summarization task using the XSum (Extreme Summarization) dataset.

To fine-tune the model, I use LoRA, or Low Rank Adaptation, from the PEFT (Parameter Efficient Fine-Tuning) family. It is a technique that accelerates the fine-tuning of large models while consuming less memory, which is something valuable in my Google Colab environment! The idea is to freeze the original pre-trained weights and introduce an additional matrix computed using matrix decomposition, which makes it small. This new matrix is trained on new data while the original weights matrix are frozen, thus reducing the number of trainable parameters. Finally, both the original and the adapted weights are combined.

I chose to work on summarization as it is a task we all need from time to time, or even daily. I will be using the [XSum](https://arxiv.org/pdf/1808.08745) dataset available on [Hugging Face](https://huggingface.co/datasets/EdinburghNLP/xsum), which contains BBC articles accompanied with single-sentence summaries. I found the extreme summarization process interesting and fun. The dataset is preprocessed using the `AutoTokenizer` from Hugging Face, which makes the tokens and the vocabulary corresponds to the T5 architecture and its pretraining. The sentences are also truncated and padded to the maximum input length (here, 1024 tokens). Only 1 epoch will be run on this dataset for the fine-tuning, because of the lack of GPU. 

### 3. Quantization (Bonus, Based on Need)

To quantize the model, we can perform a QLoRA fine-tuning, which combine a LoRA fine-tuning with quantization. However, this technique implies a second GPU run, so this technique has not been done because of the lack of free GPU. Moreover, it is not necessary with a T5-small model, as it is fast enough to get a response in a reasonable delay.

### 4. Model Evaluation (Bonus)

To assess the performance of the fine-tuned model compared to the original model, I chose the ROUGE metric. [ROUGE](https://huggingface.co/spaces/evaluate-metric/rouge), or Recall-Oriented Understudy for Gisting Evaluation, is used for evaluating automatic summarization and machine translation software in Natural Language Processing (NLP). In our case, it will compare an automatically produced summary with a human-produced (set of) reference(s). ROUGE is case insensitive and consists of 4 different metrics (based on unigrams, bigrams, longest common subsequences, or lines).

### 5. API Creation

I used Flask to create a demonstration for the summarization task using the fine-tuned T5-small model. It is a Python librairy that allows the user to create web apps easily. It provides a simple and intuitive interface for creating and deploying demos.

In this app, one will be able to prompt a text in a box and get its summary from the fine-tuned T5-small model. It is run using the Python script and accessed from a Web browser.

The application is not only building a front-end but also manage the back-end and the Hugging Face model handling (loading, tokenizing, generating).

Some unit tests have been written to check the proper operation of the application (proper home page, proper generation, empty input handling). To run these tests, in the `./app/` folder, run the following command:
```bash
pytest
```

### 6. Containerization

I provide a Dockerfile to encapsulate the entire application (Flask API for text summarization) for ease of deployment. The created container will contain everything needed to properly run the application.

Hence, the deployment of the model and the application is automated and all requirements are provided in the container.

### 7. CI/CD Pipeline with GitHub Actions

The Python files of this repository have been written using Flake8 linting tool on VSCode.

I add some GitHub Actions workflows to focus on continuous integration:
● For Code Quality Checks: I used the Pylint linting tool ;
● For Docker Build: I automate the building of the Docker image *(but I have this error: `no space left on device`)* ;
● For API Testing: I used Python application test workflow.
