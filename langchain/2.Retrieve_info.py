import os
from langchain_openai import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv, find_dotenv
from config.conf import username, password, uri

load_dotenv(find_dotenv(), override=True)
os.environ["NEO4J_URI"] = uri
os.environ["NEO4J_USERNAME"] = username
os.environ["NEO4J_PASSWORD"] = password

graph = Neo4jGraph(
    url=os.environ["NEO4J_URI"], username=os.environ["NEO4J_USERNAME"], password=os.environ["NEO4J_PASSWORD"])
print(graph.schema)

chain = GraphCypherQAChain.from_llm(ChatOpenAI(temperature=0), graph=graph, verbose=True)

graph_result_1 = chain.invoke("Who was married with Albert Einstein?")
print(graph_result_1)
graph_result_2 = chain.invoke("Who were the siblings of Leonhard Euler?")
print(graph_result_2)
