import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def plot_2d_geo(G: nx.Graph, pos: dict) -> None:
    """Plot a 2D geometric graph.

    Args:
        G: A networkx Graph object.
        pos: A dictionary mapping node IDs to (x, y) coordinates.
    """
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    nx.draw_networkx_nodes(G, pos, node_size=80)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Random geometric graph")
    plt.subplot(1, 2, 2)
    # Get the degrees of all nodes
    degrees = [d for n, d in G.degree()]
    plt.hist(degrees, bins=np.arange(min(degrees), max(degrees) + 1) - 0.5, density=True, alpha=0.75)
    plt.title("Degree Distribution Histogram")
    plt.xlabel("Degree")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.show()




def plot_spring_layout(G: nx.Graph) -> None:
    """Plot a spring layout graph.

    Args:
        G: A networkx Graph object.
    """
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    pos = nx.spring_layout(G) 
    nx.draw_networkx_nodes(G, pos, node_size=70)
    nx.draw_networkx_edges(G, pos, width=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    # Get the degrees of all nodes
    degrees = [d for n, d in G.degree()]
    plt.hist(degrees, bins=np.arange(min(degrees), max(degrees) + 1) - 0.5, density=True, alpha=0.75)
    plt.title("Degree Distribution Histogram")
    plt.xlabel("Degree")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.show()
    plt.show()
        
def plot_adjacenty(trueGraph, preGraphs=[], campThis= "viridis"):
    maxValue= 0

    for preGraph in preGraphs:
        maxTemp= np.max(preGraph)
        maxValue= maxTemp if maxTemp>maxValue else maxValue

    plt.figure(figsize=(6*len(preGraphs), 6))
    plt.subplot(1, 5, 1)
    plt.imshow(trueGraph, cmap= campThis, norm=Normalize(vmin=0, vmax= maxValue))
    plt.title('Real graph')

    for i in range(len(preGraphs)):
        pass
