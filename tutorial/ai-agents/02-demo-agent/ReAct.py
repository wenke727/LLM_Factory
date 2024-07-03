# %%
from dotenv import load_dotenv

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents.tools import Tool
from langchain.agents import AgentExecutor
from langchain.agents import create_react_agent

from langchain_community.utilities import SerpAPIWrapper

load_dotenv("../.env")


# %%
# 从hub中获取React的Prompt
prompt = hub.pull("hwchase17/react")
prompt.pretty_print()

# %%
# 选择要使用的LLM
llm = ChatOpenAI(model='gpt-4o')
llm

# %%
# 实例化SerpAPIWrapper
search = SerpAPIWrapper()

# 准备工具列表
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="当大模型没有相关知识时，用于搜索知识",
    ),
]
tools

# %%
# 构建ReAct代理
agent = create_react_agent(llm, tools, prompt)
agent

# %%
# 创建代理执行器并传入代理和工具
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor

# %%
# 调用代理执行器，传入输入数据
print("第一次运行的结果：")
agent_executor.invoke({"input": "当前Agent最新研究进展是什么?"})

# %%
print("第二次运行的结果：")
agent_executor.invoke({"input": "当前Agent最新研究进展是什么?"})

# %%
