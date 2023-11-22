import torch
import networkx as nx
import random



def get_ER_random_contact(n: int, avgDegree: float, device: str="cpu") -> torch.Tensor:
    """
    Generates a random contact matrix for an Erdős-Rényi random graph.

    Args:
        n: The number of nodes in the graph.
        avgDegree: The average degree of the graph.
        device: The device on which to store the contact matrix.

    Returns:
        contact: A torch.Tensor of shape (n, n) representing the contact matrix.
        graph: networkx graph object
    """

    graph = nx.dense_gnm_random_graph(n, n * avgDegree)
    while nx.is_connected(graph) is False:
        graph = nx.dense_gnm_random_graph(n, n * avgDegree)
    contact = torch.FloatTensor(nx.to_numpy_array(graph)).to(device)
    return contact, graph

def get_WS_random_contact(n: int, k: int, p: float, device: str="cpu") -> torch.Tensor:
    """
    Generates a random contact matrix for a Watts-Strogatz small-world graph.

    Args:
        n: The number of nodes in the graph.
        k: The number of nearest neighbors to connect to.
        p: The probability of rewiring each edge.
        device: The device on which to store the contact matrix.

    Returns:
        contact: A torch.Tensor of shape (n, n) representing the contact matrix.
        graph: networkx graph object
    """

    graph = nx.watts_strogatz_graph(n, k, p)
    while nx.is_connected(graph) is False:
        graph = nx.watts_strogatz_graph(n, k, p)
    contact = torch.FloatTensor(nx.to_numpy_array(graph)).to(device)
    return contact, graph

def get_BA_random_contact(n: int, m: int, device: str="cpu") -> torch.Tensor:
    """
    Generates a random contact matrix for a Barabási-Albert preferential attachment graph.

    Args:
        n: The number of nodes in the graph.
        m: The number of edges to attach from a new node to existing nodes.
        device: The device on which to store the contact matrix.

    Returns:
        contact: A torch.Tensor of shape (n, n) representing the contact matrix.
        graph: networkx graph object
    """
    graph = nx.barabasi_albert_graph(n, m)
    while nx.is_connected(graph) is False:
        graph = nx.barabasi_albert_graph(n, m)
    contact = torch.FloatTensor(nx.to_numpy_array(graph)).to(device)
    return contact, graph

def get_Geo_random_contact(n: int, radius: float, device: str="cpu") -> torch.Tensor:
    """
    Generates a random contact matrix for a random geometric graph.

    Args:
        n: The number of nodes in the graph.
        radius: The radius of creating a link.
        device: The device on which to store the contact matrix.

    Returns:
        contact: A torch.Tensor of shape (n, n) representing the contact matrix.
        graph: networkx graph object
        pos: the 2-d position of all nodes
    """

    pos = {i: (random.uniform(0, 1), random.uniform(0, 1)) for i in range(n)}
    graph = nx.random_geometric_graph(n, radius, dim=2, pos=pos)
    contact = torch.FloatTensor(nx.to_numpy_array(graph)).to(device)
    return contact, graph, pos
