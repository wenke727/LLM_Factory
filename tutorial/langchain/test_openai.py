#%%
import os
from dotenv import load_dotenv
# from openai import OpenAI

import base64
from mimetypes import guess_type
from urllib.parse import urlparse
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv('.env')


def is_url(image_path):
    # Check if the input string is a valid URL
    try:
        result = urlparse(image_path)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def append_image_to_content(content, image_path):
    # Check if image_path is a URL or a local file path
    if is_url(image_path):
        img_url = image_path
    else:
        img_url = local_image_to_data_url(image_path)

    # Append the new image to the content list
    content.append({
        "type": "image_url",
        "image_url": {
            "url": img_url
        }
    })

client = ChatOpenAI(model = 'gpt-4o')

# res = client.invoke(['hello, i‘m bob'])
# res.pretty_print()


#%%
msg = HumanMessage(content="Describe the two picture, 还有他们两之间的关系")
msg.content

msg.pretty_print()

#%%



img = local_image_to_data_url("./honor.jpg")

content = [
    {"type": "text", "text": "Describe the two picture, 还有他们两之间的关系"},
    {   "type": "image_url",
        "image_url": {
            "url": "https://www.google.com.hk/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png"
        }
    },
    {
        "type": "image_url",
        "image_url": {
            "url": img
        }
    },
]

msgs = [HumanMessage(content = content)]

ans = client.invoke(msgs, max_tokens=100)
ans.pretty_print()

# %%
