"""
The Python code shows how to use Gradio to create a demo for a text-generation model trained using transformers. The code allows users to input a text prompt and generate a continuation of the text.

Gradio is a Python library that allows you to create and share interactive machine learning demos with ease. It provides a simple and intuitive interface for creating and deploying demos, and it supports a wide range of machine learning frameworks and libraries, including transformers.

This Python script is a Gradio demo for a text chatbot. It uses the LoRA fine-tuned T5-small model to generate summary to user input.
"""

from typing import Iterator
import gradio as gr
from transformers.utils import logging

# Load the model
# from model import get_input_token_length, run


########################################## INFERENCE PART ##########################################

DEFAULT_SYSTEM_PROMPT = """"""
MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = 4000

DESCRIPTION = """This application allows you to summarize English texts and get a very small text as
                 output. It is using a T5 small model for summarization task fine-tuned with LoRA
                 using the XSum dataset."""
LICENSE = ""

logging.set_verbosity_info()
logger = logging.get_logger("transformers")
logger.info("Starting")

def clear_and_save_textbox(message: str) -> tuple[str, str]:
    """This function clears the textbox and returns the input message to the
    saved_input state variable."""
    return '', message

def display_input(message: str,
                  history: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """This function displays the input message in the chatbot and adds the message to
    the chat history."""
    history.append((message, ''))
    logger.info("display_input=%s", message)             
    return history

def delete_prev_fn(history: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], str]:
    """This function deletes the previous response from the chat history and returns the
    updated chat history and the previous response."""
    try:
        message, _ = history.pop()
    except IndexError:
        message = ''
    return history, message or ''

def generate(message: str,
             history_with_input: list[tuple[str, str]],
             system_prompt: str,
             max_new_tokens: int,
             temperature: float,
             top_p: float,
             top_k: int) -> Iterator[list[tuple[str, str]]]:
    """This function generates text using the fine-tuned T5 small model and the given
    parameters. It returns an iterator that yields a list of tuples, where each tuple
    contains the input message and the generated response."""
    logger.info("message=%s",message)
    # if max_new_tokens > MAX_MAX_NEW_TOKENS:
    #     raise ValueError

    # history = history_with_input[:-1]
    # generator = run(message, history, system_prompt,
    #                 max_new_tokens, temperature, top_p, top_k)
    # try:
    #     first_response = next(generator)
    #     yield history + [(message, first_response)]
    # except StopIteration:
    #     yield history + [(message, '')]
    # for response in generator:
    #     yield history + [(message, response)]
    for x in ["Test", "of", "generator"]:
        yield history_with_input + [(message, x)]

def process_example(message: str) -> tuple[str, list[tuple[str, str]]]:
    """This function generates a response to the given message and returns the empty
    string and the generated response."""
    generator = generate(message, [], DEFAULT_SYSTEM_PROMPT, 1024, 1, 0.95, 50)
    for x in generator:
        pass
    return '', x

def check_input_token_length(message: str,
                             chat_history: list[tuple[str, str]],
                             system_prompt: str) -> None:
    pass
    # #logger.info("check_input_token_length=%s", message)
    # input_token_length = get_input_token_length(message, chat_history, system_prompt)
    # #logger.info("input_token_length", input_token_length)
    # #logger.info("MAX_INPUT_TOKEN_LENGTH", MAX_INPUT_TOKEN_LENGTH)
    # if input_token_length > MAX_INPUT_TOKEN_LENGTH:
    #     logger.info("Inside IF condition")
    #     raise gr.Error(f'The input is too long ({input_token_length} > {MAX_INPUT_TOKEN_LENGTH}).'
    #                    'Check the length of your document and try again.')
    # #logger.info("End of check_input_token_length function")


########################################## APPLICATION PART ##########################################

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(value='Duplicate Space for private use', elem_id='duplicate-button')
 
    with gr.Group():
        chatbot = gr.Chatbot(label='Summarizer')
        with gr.Row():
            textbox = gr.Textbox(container=False, show_label=False,
                                 placeholder='Type a message...', scale=10)
            submit_button = gr.Button('Submit', variant='primary', scale=1, min_width=0)
    with gr.Row():
        retry_button = gr.Button('Retry', variant='secondary')
        undo_button = gr.Button('Undo', variant='secondary')
        clear_button = gr.Button('Clear', variant='secondary')
 
    saved_input = gr.State()
 
    with gr.Accordion(label='Advanced options', open=False):
        system_prompt = gr.Textbox(label='System prompt', value=DEFAULT_SYSTEM_PROMPT, lines=6)
        max_new_tokens = gr.Slider(label='Max new tokens', minimum=1,
                                   maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS)
        temperature = gr.Slider(label='Temperature', minimum=0.1, maximum=4.0, step=0.1, value=1.0)
        top_p = gr.Slider(label='Top-p (nucleus sampling)',
                          minimum=0.05, maximum=1.0, step=0.05, value=0.95)
        top_k = gr.Slider(label='Top-k', minimum=1, maximum=1000, step=1, value=50)
 
    gr.Markdown(LICENSE)
 
    textbox.submit(fn=clear_and_save_textbox,
                   inputs=textbox, outputs=[textbox, saved_input], api_name=False, queue=False
    ).then(fn=display_input,
           inputs=[saved_input, chatbot], outputs=chatbot, api_name=False, queue=False
    ).then(fn=check_input_token_length,
           inputs=[saved_input, chatbot, system_prompt], api_name=False, queue=False
    ).success(fn=generate,
              inputs=[saved_input, chatbot, system_prompt, max_new_tokens, temperature, top_p, top_k],
              outputs=chatbot, api_name=False)
 
    button_event_preprocess = submit_button.click(fn=clear_and_save_textbox,
                                                  inputs=textbox, outputs=[textbox, saved_input],
                                                  api_name=False, queue=False
    ).then(fn=display_input,
           inputs=[saved_input, chatbot], outputs=chatbot, api_name=False, queue=False
    ).then(fn=check_input_token_length,
           inputs=[saved_input, chatbot, system_prompt], api_name=False, queue=False
    ).success(fn=generate,
              inputs=[saved_input, chatbot, system_prompt, max_new_tokens, temperature, top_p, top_k],
              outputs=chatbot, api_name=False)
 
    retry_button.click(fn=delete_prev_fn,
                       inputs=chatbot, outputs=[chatbot, saved_input], api_name=False, queue=False
    ).then(fn=display_input,
           inputs=[saved_input, chatbot], outputs=chatbot, api_name=False, queue=False
    ).then(fn=generate,
           inputs=[saved_input, chatbot, system_prompt, max_new_tokens, temperature, top_p, top_k,],
           outputs=chatbot, api_name=False)
 
    undo_button.click(fn=delete_prev_fn,
                      inputs=chatbot, outputs=[chatbot, saved_input], api_name=False, queue=False
    ).then(fn=lambda x: x,
           inputs=[saved_input], outputs=textbox, api_name=False, queue=False)
 
    clear_button.click(fn=lambda: ([], ''),
                       outputs=[chatbot, saved_input], api_name=False, queue=False)
 
demo.queue(max_size=20).launch(share=False, server_name="127.0.0.1")