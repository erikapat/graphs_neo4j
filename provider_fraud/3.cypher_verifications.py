
# Give me a cypher code that gives me the provider with more physician code: 65 physicians

MATCH (p:provider)-[:WORK_WITH]->(pp:physician)
WITH p, COUNT(pp) AS num_physicians
ORDER BY num_physicians DESC
LIMIT 1
MATCH (p)-[r:WORK_WITH]->(pp:physician)
RETURN p, r, pp
