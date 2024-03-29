from neo4j import GraphDatabase
import pandas as pd

# Define Neo4j connection details
from config.conf import username, password, database, uri


# Read data from Parquet file into a DataFrame
df = pd.read_parquet('output_data/claim_features.parquet').sample(10000)
print(df.shape)

# Establish connection to Neo4j
driver = GraphDatabase.driver(uri, auth=(username, password))

# Erase existing nodes and relationships
def erase_data(tx):
    tx.run("MATCH (n) DETACH DELETE n")

# Upload data to Neo4j and create nodes and relationships
def upload_data(tx):
    # Create Provider nodes and AttendingPhysician nodes
    for _, row in df.iterrows():
        provider_id = row['Provider']
        physician_id = row['AttendingPhysician']
        tx.run("""
        MERGE (p:Provider {id: $provider_id})
        MERGE (a:AttendingPhysician {id: $physician_id})
        """, provider_id=provider_id, physician_id=physician_id)

    # Create relationships between providers and attending physicians
    for provider_id, group in df.groupby('Provider'):
        for physician_id in group['AttendingPhysician']:
            tx.run("""
            MATCH (p:Provider {id: $provider_id}), (a:AttendingPhysician {id: $physician_id})
            MERGE (p)-[:WORKS_WITH]->(a)
            """, provider_id=provider_id, physician_id=physician_id)

# Erase existing data in Neo4j
with driver.session(database=database) as session:
    session.write_transaction(erase_data)

# Upload data to Neo4j
with driver.session(database=database) as session:
    session.write_transaction(upload_data)

# Close the connection
driver.close()