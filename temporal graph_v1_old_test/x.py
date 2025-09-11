
import networkx as nx
print(nx.__file__)           # Should point to site-packages/networkx
print(hasattr(nx, 'read_gpickle'))  # Should print True