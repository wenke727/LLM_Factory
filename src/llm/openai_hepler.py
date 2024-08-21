import io
import os
import base64
import openai
from PIL import Image
from loguru import logger
from mimetypes import guess_type
from dotenv import load_dotenv
from urllib.parse import urlparse

from langchain_openai import (
    OpenAI,
    ChatOpenAI,
    AzureOpenAI,
    AzureChatOpenAI
)
from langchain_core.messages import HumanMessage


def local_image_to_data_url(image_path, max_size_kb=90):
    """
    Convert a local image file to a data URL with a size limit.

    Args:
        image_path (str): The path to the image file.
        max_size_kb (int): Maximum size of the image in kilobytes.

    Returns:
        str: A data URL string representing the image.
    """
    # Guess the MIME type based on file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'

    # Open the image and check its size
    with Image.open(image_path) as img:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        size_kb = len(buffer.getvalue()) / 1024

        # If the image is too large, resize it
        if size_kb > max_size_kb:
            scale_factor = (max_size_kb / size_kb) ** 0.5
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)
            img = img.resize((new_width, new_height), Image.ANTIALIAS)

            # Save the resized image to the buffer
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")

        # Encode image file to base64
        base64_encoded_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

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

def initialize_openai_client():
    """
    Initialize the OpenAI client based on the available environment variables.

    Returns:
        completions: The completion client object.
        embeddings: The embeddings client object.
        chat: The chat completion create function (if applicable).
    """
    openai_version = int(openai.__version__.split(".")[0])

    if openai_version == 0:
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        client = None
        chat = openai.ChatCompletion.create
        embeddings = openai.Embedding.create
    else:
        azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        if azure_api_key:
            client = openai.AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("OPENAI_API_VERSION"),
            )
        else:
            client = openai.OpenAI()

        chat = client.chat.completions.create
        embeddings = client.embeddings.create

    # Wrap chat and embeddings with logging
    # chat = log_function_call(chat)
    # embeddings = log_function_call(embeddings)

    return client, chat, embeddings

def initialize_ai_client(model, use_chat_client=True, *args, **kwargs):
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
        logger.info("Initializing Azure OpenAI client...")
        # FIXME
        params = {
            "model":  os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", model),
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

        logger.info("Initializing OpenAI client...")
        if use_chat_client:
            return ChatOpenAI(api_key=openai_api_key, model=model, *args, **kwargs)
        else:
            return OpenAI(api_key=openai_api_key, model=model, *args, **kwargs)


if __name__ == "__main__":
    load_dotenv("./.env", verbose=True)

    """ OpenAI Version """
    client, chat, embeddings = initialize_openai_client()
    # chat
    response = chat(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello, world!"}])
    response.choices[0].message.content
    # embeddings
    embedding = embeddings(input="Hello, world!", model='text-embedding-3-large')
    embedding.data[0].embedding


    """ Langchain Version """
    # client = ChatOpenAI(model = 'gpt-4o')
    client = initialize_ai_client(use_chat_client=True, model='gpt-4o', temperature=0.01)

    # text
    res = client.invoke(["hi, i'm bob"])
    res.pretty_print()

    # multi-modality, 支持图片输入，也支持本地图片输入
    query = create_multimodal_input(
        text = "描述图片",
        imgs = ['https://www.google.com.hk/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png',]
    )
    ans = client.invoke([query], max_tokens=100)
    ans.pretty_print()
