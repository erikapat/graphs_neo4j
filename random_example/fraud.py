from neo4j import GraphDatabase
import pandas as pd

# Define Neo4j connection details
from config.conf import username, password, database, uri

# Define a function to initialize or erase the graph
def initialize_or_erase_graph(driver):

    # Define a session to execute Cypher queries
    with driver.session() as session:
        # Execute a Cypher query to erase all nodes and relationships from the graph
        session.run("MATCH (n) DETACH DELETE n")

    # Close the driver connection
    driver.close()


# Function to upload nodes to Neo4j
def upload_nodes(tx, nodes):
    for node in nodes:
        label = node.get('label', '')  # Retrieve the label from the node dictionary
        color = node.get('color', '')  # Retrieve the color from the node dictionary
        properties = {k: v for k, v in node.items() if k != 'label' and k != 'Nodecolor'}  # Exclude the 'label' and 'color' keys
        query = f"CREATE (n:{label} $props)"
        tx.run(query, props={**properties, 'color': color})

# Function to upload edges to Neo4j
def upload_edges(tx, edges):
    for edge in edges:
        source_id = edge.get('source_id')
        target_id = edge.get('target_id')
        relationship_type = edge.get('relationship_type')
        query = f"""
        MATCH (source), (target)
        WHERE source.node_id = $source_id AND target.node_id = $target_id
        CREATE (source)-[:{relationship_type}]->(target)
        """
        tx.run(query, source_id=source_id, target_id=target_id)

# --------------------------------------------------------------------------------------------------------
# Read data from Excel file
nodes_df = pd.read_excel('fraude.xlsx', sheet_name='nodes')
edges_df = pd.read_excel('fraude.xlsx', sheet_name='edges')

# Convert DataFrames to lists of dictionaries
nodes_data = nodes_df.to_dict(orient='records')
edges_data = edges_df.to_dict(orient='records')

# Establish connection to Neo4j
driver = GraphDatabase.driver(uri, auth=(username, password), database=database)

# Call the function to initialize or erase the graph
initialize_or_erase_graph(driver)

# Upload nodes and edges to Neo4j
with driver.session() as session:
    session.execute_write(upload_nodes, nodes_data)
    session.execute_write(upload_edges, edges_data)

# Close the connection
driver.close()
