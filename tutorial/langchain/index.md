---
sidebar_position: 0
sidebar_class_name: hidden
---
# Tutorials

New to LangChain or to LLM app development in general? Read this material to quickly get up and running.

### Basics
- [x] [Build a Simple LLM Application with LCEL](./tutorials/llm_chain.ipynb)
- [x] [Build a Chatbot](./tutorials/chatbot.ipynb)
- [x] [Build vector stores and retrievers](./tutorials/retrievers.ipynb)
- [x] [Build an Agent](./tutorials/agents.ipynb)

### Working with external knowledge
- [Build a Retrieval Augmented Generation (RAG) Application](/docs/tutorials/rag)
- [Build a Conversational RAG Application](/docs/tutorials/qa_chat_history)
- [Build a Question/Answering system over SQL data](/docs/tutorials/sql_qa)
- [Build a Query Analysis System](/docs/tutorials/query_analysis)
- [Build a local RAG application](/docs/tutorials/local_rag)
- [Build a Question Answering application over a Graph Database](/docs/tutorials/graph)
- [Build a PDF ingestion and Question/Answering system](/docs/tutorials/pdf_qa/)

### Specialized tasks
- [Build an Extraction Chain](/docs/tutorials/extraction)
- [Generate synthetic data](/docs/tutorials/data_generation)
- [Classify text into labels](/docs/tutorials/classification)
- [Summarize text](/docs/tutorials/summarization)

### LangGraph

LangGraph is an extension of LangChain aimed at
building robust and stateful multi-actor applications with LLMs by modeling steps as edges and nodes in a graph.

LangGraph documentation is currently hosted on a separate site.
You can peruse [LangGraph tutorials here](https://langchain-ai.github.io/langgraph/tutorials/).

### LangSmith

LangSmith allows you to closely trace, monitor and evaluate your LLM application.
It seamlessly integrates with LangChain, and you can use it to inspect and debug individual steps of your chains as you build.

LangSmith documentation is hosted on a separate site.
You can peruse [LangSmith tutorials here](https://docs.smith.langchain.com/tutorials/).

For a longer list of tutorials, see our [cookbook section](https://github.com/langchain-ai/langchain/tree/master/cookbook).

# load env

```python
from dotenv import load_dotenv
load_dotenv('../.env', verbose=True)
```
