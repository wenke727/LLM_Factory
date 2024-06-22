import base64
from mimetypes import guess_type
from urllib.parse import urlparse
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
