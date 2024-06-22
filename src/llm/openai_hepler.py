import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAI
from langchain_openai import AzureChatOpenAI, AzureOpenAI


def initialize_ai_client(use_chat_client=False):
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
        if use_chat_client:
            return AzureChatOpenAI(endpoint=azure_endpoint, api_key=azure_api_key)
        else:
            return AzureOpenAI(endpoint=azure_endpoint, api_key=azure_api_key)
    else:
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables.")
        print("Initializing OpenAI client...")
        if use_chat_client:
            return ChatOpenAI(api_key=openai_api_key)
        else:
            return OpenAI(api_key=openai_api_key)

if __name__ == "__main__":
    load_dotenv("./.env", verbose=True)
    client = initialize_ai_client(use_chat_client=True)

    res = client.invoke(["hi, i'm bob"])
    res.pretty_print()
    print(res.response_metadata)

