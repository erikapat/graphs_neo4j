import infomap
import networkx as nx

def findCommunities(G):
    """
    Partition network with the Infomap algorithm.
    Annotates nodes with 'community' id and return number of communities found.
    """
    im = infomap.Infomap(two_level=True, silent=True)

    print("Building Infomap network from a NetworkX graph...")
    for e in G.edges():
        im.addLink(*e)

    print("Find communities with Infomap...")
    im.run();

    print(f"Found {im.num_top_modules} modules with codelength {im.codelength:.8f} bits")

    communities = {}
    for node, module in im.modules:
        communities[node] = module

    nx.set_node_attributes(G, communities, 'community')

    return G

def drawNetwork(G):
    # position map
    pos = nx.spectral_layout(I)
    # community ids
    communities = [v for k,v in nx.get_node_attributes(I, 'community').items()]
    numCommunities = max(communities) + 1
    # color map from http://colorbrewer2.org/
    cmapLight = colors.ListedColormap(['#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6'], 'indexed', numCommunities)
    cmapDark = colors.ListedColormap(['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a'], 'indexed', numCommunities)

    # Draw edges
    nx.draw_networkx_edges(I, pos)

    # Draw nodes
    nodeCollection = nx.draw_networkx_nodes(I,
      pos = pos,
      node_size=1,
      node_color = communities,
      cmap = cmapLight
    )
    # Set node border color to the darker shade
    darkColors = [cmapDark(v) for v in communities]
    nodeCollection.set_edgecolor(darkColors)

    plt.axis('off')
    plt.xlim(np.vstack(list(pos.values()))[:, 0].min() + 0.01, np.vstack(list(pos.values()))[:, 0].max())
    plt.ylim(np.vstack(list(pos.values()))[:, 1].min(), np.vstack(list(pos.values()))[:, 1].max())
    plt.savefig('fig/draw_graph.png')
    plt.show()