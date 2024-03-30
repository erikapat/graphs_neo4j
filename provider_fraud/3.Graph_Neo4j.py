from neo4j import GraphDatabase
import pandas as pd
import time
# Define Neo4j connection details
from config.conf import username, password, database, uri
from py2neo import Graph

# ----------------------------------------------------------------------------------------------
# Consider use
# %pip install graphdatascience
# gds = GraphDataScience(connectionUrl, auth=(username, password))
# gds.version()
# ---------------------------------------------------------------------------------------------

# Connect to Neo4j and specify the database
graph = Graph(uri=uri, auth=(username, password), name=database)

def time_cypher(cypher_query):
    """

    :param cypher_query:
    :return:
    """
    start_time = time.time()
    graph.run(cypher_query)
    end_time = time.time()

    # Calcular el tiempo transcurrido
    elapsed_time = end_time - start_time
    # print("Tiempo transcurrido:", elapsed_time, "segundos")
    print("Tiempo transcurrido:", round(elapsed_time/60, 2), "minutes")


df_work_with = pd.read_csv('output_data/df_work_with.csv')
physician_nodes_df = pd.read_csv('output_data/physician_nodes_df.csv')
provider_nodes_df = pd.read_csv('output_data/provider_nodes_df.csv')


# Erase all/ Initialize
cypher_query = """
MATCH (n) DETACH DELETE n;
"""
# Execute the Cypher query to create the node and insert it into the database
time_cypher(cypher_query)

cypher_query = """
DROP CONSTRAINT provider_id IF EXISTS;
"""
time_cypher(cypher_query)

cypher_query = """
DROP CONSTRAINT physician_id IF EXISTS;
"""

time_cypher(cypher_query)

cypher_query = """
CREATE CONSTRAINT physician_id IF NOT EXISTS
FOR (x:physician)
REQUIRE x.physician_id IS UNIQUE;
"""

time_cypher(cypher_query)

cypher_query = """
CREATE CONSTRAINT provider_id IF NOT EXISTS
FOR (x:provider)
REQUIRE x.provider_id IS UNIQUE;
"""

time_cypher(cypher_query)

print('provider')

cypher_query = """
LOAD CSV WITH HEADERS
FROM 'file:///provider_nodes_df.csv' AS row
MERGE (p:provider {providerId: toInteger(row.provider_id)})
SET
p.category_id = row.category_id,
p.PotentialFraud = toInteger(row.PotentialFraud)

"""
time_cypher(cypher_query)

print('physician')

cypher_query = """
CALL apoc.periodic.iterate(
  'LOAD CSV WITH HEADERS FROM "file:///physician_nodes_df.csv" AS row RETURN row',
  'MERGE (pp:physician {physicianId: toInteger(row.physician_id)}) SET pp.category_id = row.category_id',
  {batchSize:1000, iterateList:true, parallel:true}
)
"""

time_cypher(cypher_query)

print('edges')


# Definir la consulta Cypher
cypher_query = """
CALL apoc.periodic.iterate(
  'LOAD CSV WITH HEADERS FROM "file:///df_work_with.csv" AS row RETURN row',
  'MATCH (p:provider {providerId: toInteger(row.provider_id)})
   MATCH (pp:physician {physicianId: toInteger(row.physician_id)})
   MERGE (p)-[r:WORK_WITH]->(pp)
   SET r.role = row.ClaimID',
  {batchSize:1000, iterateList:true, parallel:true}
)
"""

# Ejecutar la consulta Cypher
time_cypher(cypher_query)


print('final')
cypher_query = """
MATCH (p:provider)-[:WORK_WITH]->()
WITH DISTINCT p SET p:provider;
"""

time_cypher(cypher_query)

print("Relaciones creadas exitosamente.")

'''

# Obtener los DataFrames de pandas
df_work_with = pd.read_csv('output_data/df_work_with.csv')

# Iterar sobre cada fila del DataFrame df_work_with
for index, row in df_work_with.iterrows():
    print(index)
    provider_id = row['provider_id']
    physician_id = row['physician_id']
    claim_id = row['ClaimID']

    # Crear la relación entre el proveedor y el médico
    graph.run(
        """
        MATCH (p:provider {providerId: $provider_id})
        MATCH (pp:physician {physicianId: $physician_id})
        MERGE (p)-[r:WORK_WITH]->(pp)
        SET r.role = $claim_id
        """,
        provider_id=provider_id,
        physician_id=physician_id,
        claim_id=claim_id
    )

print("Relaciones creadas exitosamente.")

'''