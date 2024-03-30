'''
# Give me a cypher code that gives me the provider with more physician code: 65 physicians

MATCH (p:provider)-[:WORK_WITH]->(pp:physician)
WITH p, COUNT(pp) AS num_physicians
ORDER BY num_physicians DESC
LIMIT 1
MATCH (p)-[r:WORK_WITH]->(pp:physician)
RETURN p, r, pp
'''

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


cypther_run = gds.run_cypher('''CALL apoc.meta.stats() YIELD stats RETURN stats.nodeCount AS nodeCount, stats.relCount as RelationshipCount;
''')
print(cypther_run)

# total node counts
cypther_run = gds.run_cypher('''
    CALL apoc.meta.stats()
    YIELD labels
    UNWIND keys(labels) AS nodeLabel
    RETURN nodeLabel, labels[nodeLabel] AS nodeCount
''')

print(cypther_run)

# total relationship counts
cypther_run = gds.run_cypher('''
    CALL apoc.meta.stats()
    YIELD relTypesCount
    UNWIND keys(relTypesCount) AS relationshipType
    RETURN relationshipType, relTypesCount[relationshipType] AS relationshipCount
''')

print(cypther_run)

#fraud money transfer flags
cypther_run = gds.run_cypher('MATCH(p:provider) RETURN p.PotentialFraud AS PotentialFraud, count(p) AS cnt')
print(cypther_run)

# Louvain community detection ----------------------------------------------------------------------------------
'''
Louvain is useful for exploratory analysis of communities because it uses a form of modularity scoring to split 
up the graph into hierarchical clusters. This means that your theories around fraud patterns and graph structure 
don’t need to be exact for it to provide informative communities and insights.
'''
# clear the graph if it exists beforehand
clear_graph_by_name('comm-projection')

g, _ = gds.graph.project('comm-projection', ['provider','physician'], {
    'WORK_WITH': {'orientation': 'UNDIRECTED'}
})

df = gds.louvain.write(g, writeProperty='louvainCommunityId')
g.drop()
print(df)

'''
The last query below orders communities by the count of flagged users so we can further examine some 
of the more concentrated flagged communities.
'''

print("Louvain Communities Ordered by count of Flagged Users")
cypther_run = gds.run_cypher('''
    MATCH (p:provider)
    WITH p.louvainCommunityId AS community,
        count(p) AS cnt,
        sum(p.PotentialFraud) as flaggedCount
    RETURN community,
        cnt,
        flaggedCount,
        toFloat(flaggedCount)/toFloat(cnt) AS flaggedRatio
    ORDER BY flaggedCount DESC LIMIT 100
''')

print(cypther_run)

'''
We can view communities of users and connecting identifiers in Neo4j Bloom or Neo4j Browser with queries of the form:

MATCH (p:provider)-[r:WORK_WITH]->(physician:physician)
WHERE p.louvainCommunityId = 85717
RETURN p, r, physician
'''

# This captures:
df_work_with = pd.read_csv('output_data/df_work_with.csv')
# physicians related with fraudulent acts  -> louvain community 85717
print(df_work_with[df_work_with['physician_id']== 2386747])
print(df_work_with[df_work_with['physician_id']== 2314546])

# Weakly Connected Components (WCC) -------------------------------------------------------------------
# clear the graph if it exists beforehand

g, _ = gds.graph.project('comm-projection', ['provider','physician'], {
    'WORK_WITH': {'orientation': 'UNDIRECTED'}
})


df = gds.wcc.write(g, writeProperty='wccId')
g.drop()
print(df)

# 79979 components where created, The majority of the components are of size 1 representing a component with
# just a single user, not resolved to any other.
# The max component size is 153 users, which means the largest community has 153 users.

'''
Labeling Fraud Risk User Accounts
As these communities are meant to label underlying groups of individuals, if even one flagged account is in the 
community, we will label all user accounts in the group as fraud risks:
'''


# Define the Cypher query to find fraud risk providers based on WCC ID
cypher_query = """
MATCH (p:provider)
WHERE p.PotentialFraud = 1
WITH collect(DISTINCT p.wccId) AS flaggedCommunities
MATCH (p:provider) WHERE p.wccId IN flaggedCommunities
SET p:FraudRiskProvider
SET p.fraudRisk = 1
RETURN count(p) AS NumberOfFraudRiskProviders
"""


# Execute the Cypher query
result = gds.run_cypher(cypher_query)

# Display the result
print(result)

# We had 506 fraudsters, now we have 531.

result = gds.run_cypher('''
    MATCH (p:provider) WHERE NOT p:FraudRiskProvider
    SET p.fraudRisk=0
    RETURN count(p)
''')
# We had 4904 non fraudsters, now we have 4879.
print(result)

# ----------------------------------------------------------------------
result= gds.run_cypher( '''
    MATCH (p:provider)
    WITH p.wccId AS community, count(p) AS cSize, sum(p.fraudRisk) AS cFraudSize
    WITH community, cSize, cFraudSize,
    CASE
        WHEN cSize=1 THEN ' 1'
        WHEN cSize=2 THEN ' 2'
        WHEN cSize=3 THEN ' 3'
        WHEN cSize>3 AND cSize<=10 THEN ' 4-10'
        WHEN cSize>10 AND cSize<=50 THEN '11-50'
        WHEN cSize>10 THEN '>50' END AS componentSize
    RETURN componentSize, 
        count(*) AS numberOfComponents, 
        sum(cSize) AS totalUserCount, 
        sum(cFraudSize) AS fraudUserCount 
    ORDER BY componentSize
''')
print(result)

result = gds.run_cypher('''
    MATCH (p:provider)
    WITH p.wccId AS WCC_ID, COUNT(p) AS ComponentSize
    RETURN WCC_ID, ComponentSize
''')

print(result)

# ----------------------------------------------------------------------------------------------------
'''
Part 3: Recommending Suspicious Accounts With Centrality & Node Similarity
In parts 1 & 2 we explored the graph and identified high risk fraud communities. At this stage, 
we may want to expand beyond our business logic to automatically identify other users that are suspiciously 
similar to the fraud risks already identified. Neo4j and GDS makes it simple to triage and recommend such 
suspect users in a matter of seconds. We can leverage both centrality and similarity algorithms for this.
'''

print('------------ FEATURE INGENIERING ------------------')

cypther_run = gds.run_cypher('''
   MATCH (p:provider)
    WITH p.wccId AS componentId, count(*) AS communitySize, collect(p) AS provider
    WITH communitySize, toInteger(communitySize > 1) AS partOfCommunity, provider
    UNWIND provider as p
    SET p.communitySize = communitySize
    SET p.partOfCommunity = partOfCommunity;
''')

print(cypther_run)

# clear the graph if it exists beforehand
clear_graph_by_name('features')

g, _ = gds.graph.project('features', ['provider','physician'], {
    'WORK_WITH': {'orientation': 'UNDIRECTED'}
})

gds.degree.write(g, relationshipTypes=['WORK_WITH'], writeProperty='workwithIdsDegree')
gds.pageRank.write(g, relationshipTypes=['WORK_WITH'], maxIterations=1000,
                   writeProperty='workwithPageRank')
gds.degree.mutate(g, nodeLabels=['provider', 'physician'], relationshipTypes=['WORK_WITH'], mutateProperty='physicianDegree')
g.drop()


df = gds.run_cypher('''
    MATCH(p: provider)
    RETURN p.providerId AS provider_id,
        p.wccId AS wccId,
        p.fraudRisk AS fraudRisk,
        p.workwithIdsDegree AS workwithIdsDegree,
        p.workwithPageRank AS workwithPageRank,
        p.communitySize AS communitySize,
        p.partOfCommunity AS partOfCommunity,
        p.louvainCommunityId AS louvainCommunityId,
        p.physicianDegree AS physicianDegree
''')

X = df #.drop(columns=['fraudRisk', 'fraudMoneyTransfer', 'wccId', 'guid'])

# y = df.fraudRisk - df.fraudMoneyTransfer

print(X.head())


# References -----------------------------------------------------------------------------------------

# - https://github.com/neo4j-product-examples/demo-fraud-detection-with-p2p/blob/main/fraud-detection-demo-with-p2p.ipynb
# https://neo4j.com/docs/getting-started/data-modeling/guide-data-modeling/