import os
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from config.conf import username, password, uri
os.environ["NEO4J_URI"] = uri
os.environ["NEO4J_USERNAME"] = username
os.environ["NEO4J_PASSWORD"] = password

graph = Neo4jGraph()
llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview")
llm_transformer = LLMGraphTransformer(llm=llm)

text="""
Albert Einstein, the iconic physicist whose name has become synonymous with genius, revolutionized our understanding of the universe with his groundbreaking theories.
Born on March 14, 1879, in Ulm, Germany, Einstein attended the Swiss Federal Institute of Technology in Zurich,
where he studied physics and mathematics. His job as a patent examiner in Bern provided him with the 
stability to pursue his scientific interests. In 1903, he married Mileva Marić, with whom he had two sons 
before their marriage ended in divorce. Einstein's most famous work, the theory of relativity, 
introduced revolutionary concepts about space, time, and gravity. His contributions to science 
earned him the Nobel Prize in Physics in 1921. Throughout his life, Einstein continued to inspire
generations with his intellect and humanitarian values. He passed away on April 18, 1955,
leaving behind a legacy that continues to shape modern physics and our understanding of the universe.
"""

documents = [Document(page_content=text)]
graph_documents = llm_transformer.convert_to_graph_documents(documents)
print(f"Nodes:{graph_documents[0].nodes}")
print(f"Relationships:{graph_documents[0].relationships}")

graph.add_graph_documents(graph_documents)