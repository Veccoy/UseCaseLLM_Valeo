# Practical Use Case : Fine-tune a LLM

The objective is to create a comprehensive GitHub repository that showcases my expertise in leveraging Generative AI models, with a particular focus on deploying, optimizing, and fine-tuning Large Language Models (LLMs).
The project involves selecting a pre-trained model from Hugging Face, applying advanced optimization techniques, retraining the model on a specified dataset, evaluating its performance, and seamlessly integrating the workflow into a CI/CD pipeline.

## Project Description

*Project Overview: Brief introduction and purpose of the project.*

*Implementation Explanation: Provide in-depth explanations of the choices and approaches taken in steps 1 through 5, detailing your decision-making process and implementation strategy.*

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

To quantize the model, we can perform a QLoRA fine-tuning, which combine a LoRA fine-tuning with quantization. However, this technique implies a second GPU run, so this technique has not been selected because of the lack of free GPU.

### 4. Model Evaluation (Bonus)

To assess the performance of the fine-tuned model compared to the original model, I chose the ROUGE metric. [ROUGE](https://huggingface.co/spaces/evaluate-metric/rouge), or Recall-Oriented Understudy for Gisting Evaluation, is used for evaluating automatic summarization and machine translation software in Natural Language Processing (NLP). In our case, it will compare an automatically produced summary with a human-produced (set of) reference(s). ROUGE is case insensitive and consists of 4 different metrics (based on unigrams, bigrams, longest common subsequences, or lines).

### 5. API Creation

I used Gradio to create a demo for the summarization task using the fine-tuned T5-small model. It is a Python librairy based on FastAPI that allows the user to create interactive ML apps easily. It provides a simple and intuitive interface for creating and deploying demos, and it supports a wide range of ML frameworks and libraries, including transformers.

In this app, one will be able to prompt a text in a chatbox and get the extreme summary from the fine-tuned T5-small model. It is run using the Python script and accessed from a Web browser.

*Testing: Write tests to validate the API's functionality.*

### 6. Containerization

I provide a Dockerfile ton encapsulate the 
Objective: Encapsulate the entire application, including the API, into a Docker container for
ease of deployment.
Requirements:
● Dockerfile: Create a Dockerfile that automates the setup of the environment, ensuring
that all dependencies are installed correctly.
● Containerization: Build and run the Docker image, verifying that the API functions as
expected within the container.

### 7. CI/CD Pipeline with GitHub Actions
Objective: Develop a CI/CD pipeline using GitHub Actions, focusing on continuous integration
(CI) rather than continuous deployment (CD).
Requirements:
Pipeline Stages: Implement a multi-stage pipeline that includes:
● Code Quality Checks: Use tools for linting.
● Docker Build: Automate the building of the Docker image.
● API Testing: Include a stage that verifies the API’s functionality by making test calls

## Setup Instructions

*Detailed steps for setting up the project locally, including environment setup.*

## User Guide

*Instructions on how to use the repository, including running the fine-tuned model and interacting with the API.*
