import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer


class ChatModel(AutoModelForCausalLM):
    def __init__(self, model_name, device="cuda"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).to(device)
        self.device = device

    def chat(self, query=None, history=[], verbose=False):
        """启动多轮对话，可指定初始提示和历史对话"""
        if len(history) == 0:
            self.history = [{"role": "system", "content": "You are a helpful assistant."}]
        else:
            self.history = history

        self.history.append({"role": "user", "content": query})

        # 应用对话模板
        text = self.tokenizer.apply_chat_template(
            self.history,
            tokenize=False,
            add_generation_prompt=True
        )

        # 编码输入并发送到设备
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        # 模型生成回答
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )

        # 截取新生成的tokens并解码
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # 打印模型回答并更新对话历史
        if verbose:
            logger.debug("AI: ", response)

        self.history.append({"role": "system", "content": response})

        return self.history  # 返回更新后的对话历史，以便于可能的后续用途

if __name__ == "__main__":
    # 使用示例
    model_name = "Qwen/Qwen1.5-32B-Chat"
    chat_model = ChatModel(model_name)
    history = [{"role": "system", "content": "You are a helpful assistant."}]
    prompt = "Tell me more about AI."
    updated_history = chat_model.chat(query=prompt, history=history)
