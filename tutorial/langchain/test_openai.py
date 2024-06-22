#%%
import sys
sys.path.append('../../src')
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from utils.multimodal_input import create_multimodal_input
load_dotenv('.env')



client = ChatOpenAI(model = 'gpt-4o')

msg = create_multimodal_input(text = "描述图片", imgs = ['./honor.jpg'])
msg

ans = client.invoke([msg], max_tokens=100)
ans.pretty_print()

# %%
ans.response_metadata

# %%
