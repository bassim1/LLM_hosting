import gradio as gr
import transformers
from torch import bfloat16
from threading import Thread


def promt_build(system_promt, user_inp, hist):
    prompt = f""" System Prompt: \n {system_promt} \n"""
    for pair in hist:
        prompt += f""" User Input: \n {pair[0]} \n"""
        prompt += f""" Assistant Response: \n {pair[1]} \n"""
    prompt += f""" User Input: \n {user_inp} \n Assistant Response: \n"""

    return prompt


def chat(system_prompts, user_inp, hist):
    prompt = promt_build(system_prompts, user_inp, hist)
    model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    streamer = transformers.TextIteratorStreamer(
        tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True,
        )
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_length=2048,
        do_sample=True,
        top_p=0.95,
        temperature=0.8,
        top_k=50,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    model_output = ""
    for new_text in streamer:
        model_output += new_text
        yield model_output
    return model_output


model_name = "stabilityai/StableBeluga-7B"
# Set up bits and bytes config
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16,
)
# Create model config using model name
model_config = transformers.AutoConfig.from_pretrained(model_name)
# Create model using model config
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map="auto",
)
# Create tokenizer using model config
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

description = """
A chatbot using StableBeluga-7B from stabilityai hosted locally by Sam Ahdab.
If you run into issues, hit the retry button.
"""
system_prompts = [
    "You are a useful AI.",
    "You are a 4chan fanatic, you only reply in greentext format."
    "You are the least helpful stack overflow user."
    "The world is ending in 30 seconds and your last words are \
    the following interaction with this user."
]

with gr.Blocks() as demo:
    gr.Markdown(description)
    gr.Markdown("**System Prompt**")
    dropdown = gr.Dropdown(
        choices=system_prompts,
        label="Select a system prompt or type a new one",
        value="You are a useful AI.",
        allow_custom_value=True,
    )
    chatbot = gr.ChatInterface(fn=chat, additional_inputs=[dropdown])
    demo.queue(api_open=False).launch(server_name="100.118.148.23", show_api=False)
