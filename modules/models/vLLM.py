from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from transformers.generation import GenerationConfig
import logging
import colorama
import requests
import json
from sseclient import SSEClient
from .base_model import BaseLLMModel
from ..presets import MODEL_METADATA


class VllmClient(BaseLLMModel):
    def __init__(self, model_name, user_name="", vllm_endpoints="") -> None:
        super().__init__(model_name=model_name, user=user_name)
        self.model_name = model_name
        self.user_name = user_name
        self.vllm_endpoints = vllm_endpoints

    def generation_config(self):
        return GenerationConfig.from_dict({
            "chat_format": "chatml",
            "do_sample": True,
            "eos_token_id": 151643,
            "max_length": self.token_upper_limit,
            "max_new_tokens": 512,
            "max_window_size": 6144,
            "pad_token_id": 151643,
            "top_k": 0,
            "top_p": self.top_p,
            "transformers_version": "4.33.2",
            "trust_remote_code": True,
            "temperature": self.temperature,
            })

    def _get_glm_style_input(self):
        history = [x["content"] for x in self.history]
        query = history.pop()
        logging.debug(colorama.Fore.YELLOW +
                      f"{history}" + colorama.Fore.RESET)
        assert (
            len(history) % 2 == 0
        ), f"History should be even length. current history is: {history}"
        history = [[history[i], history[i + 1]]
                   for i in range(0, len(history), 2)]
        return history, query

    def get_answer_at_once(self):
        history, query = self._get_glm_style_input()
        self.model.generation_config = self.generation_config()
        response, history = self.model.chat(self.tokenizer, query, history=history)
        return response, len(response)

    def get_answer_stream_iter(self):
        url = "{}/v1/chat/completions".format(self.vllm_endpoints)
        logging.info("call vllm endpoints: {}".format(url))
        headers = {"Content-Type": "application/json"}
        stop = ["<|end|>", "<|im_end|>", "<|endoftext|>", "<|user|>", "<|user", "<|system|>", "<|assistant|>"]
        stop += self.stop_sequence
        data = {
            "model": self.model_name,
            "messages": self.history,
            "max_tokens": self.max_generation_token,
            "stop": stop,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": True,
            "skip_special_tokens": False
        }
        logging.info("request data: {}".format(data))

        response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
        client = SSEClient(response)
        generated_text = ""
        for event in client.events():
            if event.data != '[DONE]':
                json_line = json.loads(event.data)
                try:
                    new_text = json_line['choices'][0]['delta']['content']
                    if new_text != "":
                        generated_text += new_text
                        yield generated_text
                except Exception as e:
                    logging.warning("parse error: {}".format(e))
