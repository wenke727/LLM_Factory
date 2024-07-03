from tqdm import tqdm
from loguru import logger
from typing import Optional
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from langchain.schema import Generation


class VLLMModel():
    DEFAULT_PARAMS = {
        "max_tokens": 1024,
        "temperature": 0,
        "top_p": 1.0,
    }

    def __init__(self, model_name, tensor_parallel_size=1, confidence=False) -> None:
        self.model_name = model_name
        self.confidence = confidence
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.llm = LLM(
            model = model_name,
            tensor_parallel_size = tensor_parallel_size,
            trust_remote_code = True
        )
        self.params = SamplingParams(
            max_tokens = self.DEFAULT_PARAMS["max_tokens"],
            temperature = self.DEFAULT_PARAMS["temperature"],
            # top_p=self.DEFAULT_PARAMS["top_p"],
            # use_beam_search=True,
            # best_of=5,
            logprobs = 1,
        )

    def generate(self, prompts):
        texts = []
        generations = []
        errors = []
        latencies = []

        tokenized_prompts = []
        for prompt in tqdm(prompts, desc="Tokenizing"):
            msg = [{"role": "user", "content": prompt}]
            tokenized_prompt = self.tokenizer.apply_chat_template(msg, add_generation_prompt=True)

            if len(tokenized_prompt) > 4096:
                logger.warning(f"Input is greater than 4096 tokens: {len(tokenized_prompt)}")
            tokenized_prompts.append(tokenized_prompt)

        responses = self.llm.generate(
            prompt_token_ids = tokenized_prompts,
            sampling_params = self.params,
            use_tqdm = True,
        )

        for response in responses:
            conf = None
            _output = response.outputs[0]
            if self.confidence:
                conf = {
                    "logprobs": {
                        "top_logprobs": self._process_confidence_request(_output.logprobs)
                    }
                }
            generation = Generation(
                text = _output.text.strip().replace("<|eot_id|>", ""),
                generation_info = conf
            )
            texts.append(_output.text)
            generations.append([generation])
            errors.append(None)
            latencies.append(0)


        return {'text': texts, "generations": generations, "errors": errors, "latencies": latencies}

    def _process_confidence_request(self, logprobs):
        resp = []
        for item in logprobs:
            key = list(item.keys())[0]
            curr_logprob_obj = item[key]
            resp.append({curr_logprob_obj.decoded_token: curr_logprob_obj.logprob})
        return resp

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        return 0

    def returns_token_probs(self) -> bool:
        return True

    def get_num_tokens(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt))