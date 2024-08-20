from dotenv import load_dotenv
load_dotenv("../../.env")
load_dotenv("./.env")

from loguru import logger
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import load_tools, AgentExecutor, create_react_agent

llm = ChatOpenAI(model="gpt-4o", temperature=.5)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
logger.info(f"load tools: \n{tools}")


template = """
尽你所能用中文回答以下问题，如果能力不够你可以使用以下工具：
{tools}

===
Use the following format:
- Question: the input question you must answer
- Thought: you should always think about what to do
- Action: the action to take, should be one of [{tool_names}]
- Action Input: the input to the action
- Observation: the result of the action
...(this Thought/ Action/ Action Input/ Observation can repeat `N` times)

Thought: I now know the final answer
Final Answer: the final answer to the original input question

===
Now, begin
Question: {input}
Thought: {agent_scratchpad}

"""

prompt = PromptTemplate.from_template(template)
logger.debug(prompt.pretty_repr())

agent = create_react_agent(llm, tools, prompt)
logger.debug(f"agent: \n{agent}")

agent_executor = AgentExecutor(
    agent=agent, tools=tools, handle_parsing_errors=True, verbose=True)

agent_executor.invoke(
    {"input":"目前市场上玫瑰花的一般进货价格是多少？如果我在此基础上加价5%，应该如何定价？"})

# langsmith log: https://smith.langchain.com/public/0b52070d-4cb8-4ffe-a0b6-64e35670fefc/r
