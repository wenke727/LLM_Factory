import os
import base64
from mimetypes import guess_type
from openai import AzureOpenAI

END_POINT = "https://xxx.openai.azure.com/"
API_KEY = ""
DEPLOYMENT_NAME = "GPT-4-vision"


def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    # https://learn.microsoft.com/zh-cn/azure/ai-services/openai/how-to/gpt-with-vision?tabs=python%2Csystem-assigned%2Cresource
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


class AzureOpenAIModelWrapper:
    def __init__(self, model_name):
        """
        初始化模型封装器，配置必要的API连接。

        参数:
        - model_name (str): 使用的模型名称，例如 'gpt-4'.
        - device (str, optional): 设备参数，此处作为占位符使用，实际不会应用于HTTP请求。默认为'azure'.

        Refs:
        - https://learn.microsoft.com/zh-cn/azure/ai-services/openai/chatgpt-quickstart?tabs=command-line%2Cpython-new&pivots=programming-language-python
        """
        # 从环境变量中读取API key、endpoint和deployment名称
        api_key = os.getenv('AZURE_OPENAI_API_KEY')
        api_base = os.getenv('AZURE_OPENAI_ENDPOINT')
        deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')

        api_version = '2023-12-01-preview'  # API版本可能会有更新

        # 构建基础URL
        base_url = f"{api_base}/openai/deployments/{deployment_name}/extensions"

        # 初始化AzureOpenAI客户端
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            base_url=base_url
        )
        self.model_name = model_name

    def chat(self, query=None, history=[], max_tokens=150, temperature=0.7,
             top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, verbose=False):
        """
        进行一轮对话生成，可以自定义对话生成的各种参数。

        参数:
        - query (str, optional): 用户的输入文本。如果提供，它将被添加到历史对话中。默认为None。
        - history (list of dict, optional): 对话历史列表，每个元素是一个包含角色和内容的字典。默认为一个系统角色的欢迎语。
        - max_tokens (int, optional): 生成回答时的最大令牌数。默认为150。
        - temperature (float, optional): 控制生成文本的随机性。较高的值增加文本的多样性和创造性，较低的值生成更确定、一致的文本。默认为0.7。
        - top_p (float, optional): 控制“核采样”的概率阈值，仅从累积概率和超过此阈值的最小集合中采样。默认为1.0，意味着禁用核采样。
        - frequency_penalty (float, optional): 减少生成回答中重复单词的概率。数值越高，重复越少。默认为0.0。
        - presence_penalty (float, optional): 增加或减少生成中未曾出现过的新颖内容的可能性。正值增加新奇，负值偏向常见内容。默认为0.0。
        - verbose (bool, optional): 如果为True，会打印更多的调试信息，包括API调用的详细错误和生成的文本。默认为False。

        返回:
        - tuple: 包含生成的文本回答和更新后的对话历史的元组。

        示例:
        >>> model = AzureOpenAIModelWrapper("gpt-4")
        >>> response, history = model.chat("How can I improve my Python skills?", max_tokens=100, temperature=0.9, verbose=True)
        >>> print("Response:", response)
        >>> print("Dialogue History:", history)

        Refs:
        - https://learn.microsoft.com/zh-cn/azure/ai-services/openai/gpt-v-quickstart?tabs=image%2Ccommand-line&pivots=programming-language-python
        """
        if len(history) == 0:
            history = [{"role": "system", "content": "You are a helpful assistant."}]
        if query is not None:
            history.append({"role": "user", "content": query})

        # 调用GPT-4进行对话
        try:
            response = self.client.ChatCompletion.create(
                model=self.model_name,
                messages=history,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )
            answer = response.choices[0].message.content
        except Exception as e:
            if verbose:
                print("Error during API call:", str(e))
            return str(e), history

        # 如果verbose模式，打印模型的回答
        if verbose:
            print(f"AI: {answer}")

        # 更新对话历史
        history.append({"role": "system", "content": answer})

        return answer, history

    def chat_with_image(self, text, image_url, history=[], max_tokens=2000):
        """
        向模型发送包含文本和图片的请求。

        参数:
        - text (str): 用户的文本输入。
        - image_url (str): 图片的URL。
        - history (list of dict, optional): 对话历史。
        - max_tokens (int): 生成回答时的最大令牌数。

        返回:
        - dict: 模型的响应。
        """
        messages = history + [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]}
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens
        )
        return response

    def chatcompetion():
        """
        Creates a model response for the given chat conversation.

        Args:
          messages: A list of messages comprising the conversation so far.
              [Example Python code](https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models).

          model: ID of the model to use. See the
              [model endpoint compatibility](https://platform.openai.com/docs/models/model-endpoint-compatibility)
              table for details on which models work with the Chat API.

          frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on their
              existing frequency in the text so far, decreasing the model's likelihood to
              repeat the same line verbatim.

              [See more information about frequency and presence penalties.](https://platform.openai.com/docs/guides/text-generation/parameter-details)

          function_call: Deprecated in favor of `tool_choice`.

              Controls which (if any) function is called by the model. `none` means the model
              will not call a function and instead generates a message. `auto` means the model
              can pick between generating a message or calling a function. Specifying a
              particular function via `{"name": "my_function"}` forces the model to call that
              function.

              `none` is the default when no functions are present. `auto` is the default if
              functions are present.

          functions: Deprecated in favor of `tools`.

              A list of functions the model may generate JSON inputs for.

          logit_bias: Modify the likelihood of specified tokens appearing in the completion.

              Accepts a JSON object that maps tokens (specified by their token ID in the
              tokenizer) to an associated bias value from -100 to 100. Mathematically, the
              bias is added to the logits generated by the model prior to sampling. The exact
              effect will vary per model, but values between -1 and 1 should decrease or
              increase likelihood of selection; values like -100 or 100 should result in a ban
              or exclusive selection of the relevant token.

          logprobs: Whether to return log probabilities of the output tokens or not. If true,
              returns the log probabilities of each output token returned in the `content` of
              `message`.

          max_tokens: The maximum number of [tokens](/tokenizer) that can be generated in the chat
              completion.

              The total length of input tokens and generated tokens is limited by the model's
              context length.
              [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken)
              for counting tokens.

          n: How many chat completion choices to generate for each input message. Note that
              you will be charged based on the number of generated tokens across all of the
              choices. Keep `n` as `1` to minimize costs.

          presence_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on
              whether they appear in the text so far, increasing the model's likelihood to
              talk about new topics.

              [See more information about frequency and presence penalties.](https://platform.openai.com/docs/guides/text-generation/parameter-details)

          response_format: An object specifying the format that the model must output. Compatible with
              [GPT-4 Turbo](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo) and
              all GPT-3.5 Turbo models newer than `gpt-3.5-turbo-1106`.

              Setting to `{ "type": "json_object" }` enables JSON mode, which guarantees the
              message the model generates is valid JSON.

              **Important:** when using JSON mode, you **must** also instruct the model to
              produce JSON yourself via a system or user message. Without this, the model may
              generate an unending stream of whitespace until the generation reaches the token
              limit, resulting in a long-running and seemingly "stuck" request. Also note that
              the message content may be partially cut off if `finish_reason="length"`, which
              indicates the generation exceeded `max_tokens` or the conversation exceeded the
              max context length.

          seed: This feature is in Beta. If specified, our system will make a best effort to
              sample deterministically, such that repeated requests with the same `seed` and
              parameters should return the same result. Determinism is not guaranteed, and you
              should refer to the `system_fingerprint` response parameter to monitor changes
              in the backend.

          stop: Up to 4 sequences where the API will stop generating further tokens.

          stream: If set, partial message deltas will be sent, like in ChatGPT. Tokens will be
              sent as data-only
              [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format)
              as they become available, with the stream terminated by a `data: [DONE]`
              message.
              [Example Python code](https://cookbook.openai.com/examples/how_to_stream_completions).

          stream_options: Options for streaming response. Only set this when you set `stream: true`.

          temperature: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
              make the output more random, while lower values like 0.2 will make it more
              focused and deterministic.

              We generally recommend altering this or `top_p` but not both.

          tool_choice: Controls which (if any) tool is called by the model. `none` means the model will
              not call any tool and instead generates a message. `auto` means the model can
              pick between generating a message or calling one or more tools. `required` means
              the model must call one or more tools. Specifying a particular tool via
              `{"type": "function", "function": {"name": "my_function"}}` forces the model to
              call that tool.

              `none` is the default when no tools are present. `auto` is the default if tools
              are present.

          tools: A list of tools the model may call. Currently, only functions are supported as a
              tool. Use this to provide a list of functions the model may generate JSON inputs
              for. A max of 128 functions are supported.

          top_logprobs: An integer between 0 and 20 specifying the number of most likely tokens to
              return at each token position, each with an associated log probability.
              `logprobs` must be set to `true` if this parameter is used.

          top_p: An alternative to sampling with temperature, called nucleus sampling, where the
              model considers the results of the tokens with top_p probability mass. So 0.1
              means only the tokens comprising the top 10% probability mass are considered.

              We generally recommend altering this or `temperature` but not both.

          user: A unique identifier representing your end-user, which can help OpenAI to monitor
              and detect abuse.
              [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """



if __name__ == "__main__":
    # 示例使用
    model = AzureOpenAIModelWrapper("gpt-4")
    response, history = model.chat("How can I improve my Python skills?", max_tokens=100, temperature=0.9, verbose=True)
    print("Response:", response)
    print("Dialogue History:", history)
