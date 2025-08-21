from graphviz import Digraph

# Create a directed graph
dot = Digraph(comment="CI/CD Pipeline for LLM Applications", format="png")
dot.attr(rankdir="LR", size="10")

# Nodes for stages
dot.node("VC", "Version Control\n(Code, Prompts, Datasets, Embeddings)", shape="box", style="filled", fillcolor="#f4f4f4")
dot.node("AT", "Automated Testing & Evaluation\n(Unit tests, Ground truth scoring,\nBias & Safety checks)", shape="box", style="filled", fillcolor="#e1f5fe")
dot.node("ST", "Staging Environment\n(Infra + Controlled Data)", shape="box", style="filled", fillcolor="#fff9c4")
dot.node("DEP", "Production Deployment", shape="box", style="filled", fillcolor="#c8e6c9")
dot.node("MON", "Monitoring & Observability\n(Metrics, Drift Detection, Logging)", shape="box", style="filled", fillcolor="#fce4ec")
dot.node("RT", "Retraining / Updates\n(Fine-tuning, RAG reindexing, Prompt changes)", shape="box", style="filled", fillcolor="#ffe0b2")

# AWS / tools layer
dot.node("AWS", "AWS CodePipeline / SageMaker / Lambda\nor Kubeflow / MLflow", shape="note", style="filled", fillcolor="#d1c4e9")

# Edges
dot.edges([("VC", "AT"), ("AT", "ST"), ("ST", "DEP"), ("DEP", "MON"), ("MON", "RT"), ("RT", "VC")])
dot.edge("VC", "AWS", style="dashed")
dot.edge("AT", "AWS", style="dashed")
dot.edge("ST", "AWS", style="dashed")
dot.edge("DEP", "AWS", style="dashed")
dot.edge("MON", "AWS", style="dashed")
dot.edge("RT", "AWS", style="dashed")

# Render the diagram
output_path = "ci_cd_llm_pipeline"
dot.render(output_path, cleanup=True)

output_path + ".png"
