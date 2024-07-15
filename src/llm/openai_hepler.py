import os
import base64
from mimetypes import guess_type
from dotenv import load_dotenv
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI, OpenAI
from langchain_openai import AzureChatOpenAI, AzureOpenAI
from langchain_core.messages import HumanMessage


def local_image_to_data_url(image_path):
    # Guess the MIME type based on file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'

    # Encode image file to base64
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Return data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def is_url(path):
    # Check if the string is a valid URL
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def append_image_to_content(content, image_path):
    # Determine if image_path is a URL or a local file
    if is_url(image_path):
        img_url = image_path
    else:
        img_url = local_image_to_data_url(image_path)

    # Append the image to the content list
    content.append({
        "type": "image_url",
        "image_url": {"url": img_url}
    })

def create_multimodal_input(text, imgs):
    """
    Initialize content with text
    """
    content = [{"type": "text", "text": text}]

    # Support both single and multiple images
    if isinstance(imgs, list):
        for img in imgs:
            append_image_to_content(content, img)
    else:
        append_image_to_content(content, imgs)

    # Return a new HumanMessage with the prepared content
    return HumanMessage(content=content)


def initialize_ai_client(use_chat_client=True, *args, **kwargs):
    """
    Initializes and returns an AI client based on environment configuration and client type.

    Args:
        use_chat_client (bool): If True, initializes a chat-specific client (e.g., ChatOpenAI or AzureChatOpenAI).
                                If False, initializes a general client (e.g., OpenAI or AzureOpenAI).

    Returns:
        An instance of either OpenAI, ChatOpenAI, AzureOpenAI, or AzureChatOpenAI based on the environment settings
        and the 'use_chat_client' flag.

    Raises:
        ValueError: If the required API keys are not found in the environment variables.
    """
    # Check if Azure OpenAI key and endpoint are available
    azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')

    if azure_api_key and azure_endpoint:
        print("Initializing Azure OpenAI client...")
        params = {
            "model":  os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            "deployment_name":  os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        }
        if use_chat_client:
            return AzureChatOpenAI(*args, **kwargs, **params)
        else:
            return AzureOpenAI(*args, **kwargs, **params)
    else:
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables.")
        print("Initializing OpenAI client...")
        if use_chat_client:
            return ChatOpenAI(api_key=openai_api_key, *args, **kwargs)
        else:
            return OpenAI(api_key=openai_api_key, *args, **kwargs)


if __name__ == "__main__":
    load_dotenv("./.env", verbose=True)
    client = initialize_ai_client(use_chat_client=True, temperature=0.01)

    # text prompt
    res = client.invoke(["hi, i'm bob"])
    res.pretty_print()
    print(res.response_metadata)

    # 多个图片输出
    client = ChatOpenAI(model = 'gpt-4o')
    msg = create_multimodal_input(text = "描述图片", imgs = ['https://www.google.com.hk/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png'])

    ans = client.invoke([msg], max_tokens=100)
    ans.pretty_print()
