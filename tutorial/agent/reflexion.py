#%%
from loguru import logger
from dotenv import load_dotenv
load_dotenv("../../.env", verbose=True)

import json
import datetime
from typing import Literal
from IPython.display import Image, display

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, MessageGraph, START


logger.add('./reflexion.log')

llm = ChatOpenAI(model="gpt-4-turbo-preview")

""" Construct tools """
search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper = search, max_results = 5)

""" Initial responder """
class ResponderWithRetries:
    def __init__(self, runnable, validator):
        self.runnable = runnable
        self.validator = validator

    def respond(self, state: list):
        response = []
        for attempt in range(3):
            response = self.runnable.invoke(
                {"messages": state}, {"tags": [f"attempt:{attempt}"]}
            )
            try:
                self.validator.invoke(response)
                return response
            except ValidationError as e:
                state = state + [
                    response,
                    ToolMessage(
                        content=f"{repr(e)}\n\nPay close attention to the function schema.\n\n"
                        + self.validator.schema_json()
                        + " Respond by fixing all validation errors.",
                        tool_call_id=response.tool_calls[0]["id"],
                    ),
                ]
        return response


class Reflection(BaseModel):
    missing: str = Field(description = "Critique of what is missing.")
    superfluous: str = Field(description = "Critique of what is superfluous")

class AnswerQuestion(BaseModel):
    """Answer the question. Provide an answer, reflection, and then follow up with search queries to improve the answer."""

    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: list[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )


# Extend the initial answer schema to include references.
# Forcing citation in the model encourages grounded responses
class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question. Provide an answer, reflection,

    cite your reflection with references, and finally
    add search queries to improve the answer."""

    references: list[str] = Field(
        description="Citations motivating your updated answer."
    )


sys_prompt = """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer."""

usr_prompt = """
<system>Reflect on the user's original question and the actions taken thus far. Respond using the {function_name} function.</reminder>
"""

actor_prompt_template = ChatPromptTemplate.from_messages([
    ("system", sys_prompt,),
    MessagesPlaceholder(variable_name = "messages"),
    ("user", usr_prompt),
]).partial(
    time = lambda: datetime.datetime.now().isoformat(),
)


_actor_prompt_template = actor_prompt_template.partial(
    first_instruction = "Provide a detailed ~250 word answer.",
    function_name = AnswerQuestion.__name__,
)
initial_answer_chain = _actor_prompt_template | llm.bind_tools(tools=[AnswerQuestion])
validator = PydanticToolsParser(tools=[AnswerQuestion])
first_responder = ResponderWithRetries(
    runnable = initial_answer_chain,
    validator = validator
)



""" revision """
revise_instructions = """
Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

_actor_prompt_template = actor_prompt_template.partial(
    first_instruction = revise_instructions,
    function_name = ReviseAnswer.__name__,
)
revision_chain = _actor_prompt_template | llm.bind_tools(tools = [ReviseAnswer])
revision_validator = PydanticToolsParser(tools = [ReviseAnswer])

revisor = ResponderWithRetries(runnable = revision_chain, validator = revision_validator)


# run
debug = False
if debug:
    example_question = "Why is reflection useful in AI?"
    initial = first_responder.respond([HumanMessage(content=example_question)])
    logger.debug("\n" + initial.pretty_repr())

    keys_level_0 = initial.tool_calls[0]['args'].keys() # ['answer', 'reflexion']
    logger.info(f"args keys: {list(keys_level_0)}")
    keys_level_1 = initial.tool_calls[0]['args']['reflection'].keys() # ['missing', 'superfluous', 'search_queries']
    logger.info(f"reflection keys: {list(keys_level_1)}")

if debug:
    revised = revisor.respond(
        [
            HumanMessage(content=example_question),
            initial,
            ToolMessage(
                tool_call_id=initial.tool_calls[0]["id"],
                content=json.dumps(
                    tavily_tool.invoke(
                        {"query": initial.tool_calls[0]["args"]['reflection']["search_queries"][0]}
                    )
                ),
            ),
        ]
    )

    logger.debug("\n" + revised.pretty_repr())

""" Create Tool Node """
def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""
    return tavily_tool.batch([{"query": query} for query in search_queries])

tool_node = ToolNode(
    [
        StructuredTool.from_function(run_queries, name = AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name = ReviseAnswer.__name__),
    ]
)

def build_graph():
    MAX_ITERATIONS = 5
    builder = MessageGraph()
    builder.add_node("draft", first_responder.respond)


    builder.add_node("execute_tools", tool_node)
    builder.add_node("revise", revisor.respond)
    # draft -> execute_tools
    builder.add_edge("draft", "execute_tools")
    # execute_tools -> revise
    builder.add_edge("execute_tools", "revise")

    # Define looping logic:


    def _get_num_iterations(state: list):
        i = 0
        for m in state[::-1]:
            if m.type not in {"tool", "ai"}:
                break
            i += 1
        return i


    def event_loop(state: list) -> Literal["execute_tools", "__end__"]:
        # in our case, we'll just stop after N plans
        num_iterations = _get_num_iterations(state)
        if num_iterations > MAX_ITERATIONS:
            return END
        return "execute_tools"


    # revise -> execute_tools OR end
    builder.add_conditional_edges("revise", event_loop)
    builder.add_edge(START, "draft")
    graph = builder.compile()

    return graph

graph = build_graph()
# display(Image(graph.get_graph().draw_mermaid_png()))

events = graph.stream(
    [HumanMessage(content="How should we handle the climate crisis?")],
    stream_mode="values",
)
for i, step in enumerate(events):
    logger.info(f"Step {i}")
    logger.debug("\n" + step[-1].pretty_repr())


# %%
