import pandas as pd

pd.set_option('display.width', 0)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.max_rows', 50)


from graphdatascience import GraphDataScience
from config.conf import username, password, database, uri


def clear_graph_by_name(g_name):
    if gds.graph.exists(g_name).exists:
        g = gds.graph.get(g_name)
        gds.graph.drop(g)

# Use Neo4j URI and credentials according to your setup
gds = GraphDataScience(uri, auth=(username, password), database=database)


