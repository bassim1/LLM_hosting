import gradio as gr
import transformers
from torch import bfloat16
from threading import Thread
from gradio.themes.utils.colors import Color


def main(model_name):
    # Create model config using model name
    model_config = transformers.AutoConfig.from_pretrained(model_name)
    # Create model using model config
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        config=model_config,
        device_map="auto",
    )
    # Create tokenizer using model config
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)


def promt_build(system_promt, user_inp, hist):
    prompt = f""" System Prompt: \n {system_promt} \n"""
    for pair in hist:
        prompt += f""" User Input: \n {pair[0]} \n"""
        prompt += f""" Assistant Response: \n {pair[1]} \n"""
    prompt += f""" User Input: \n {user_inp} \n Assistant Response: \n"""

    return prompt


main("stabilityai/StableBeluga-7B")
