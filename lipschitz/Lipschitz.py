import scipy as scp
import random, math
import networkx as nx
import numpy as np

def getLayers(model):
    return sum(1 for _ in model.children())

def getProdOfLayerWeights(model, layers):
    val = 1
    weights = [layer for layer in model.children()]
    for i in range(0, layers):
        layer_weights = weights[i].lin.weight.data.detach().cpu().numpy()
        val *= np.linalg.norm(layer_weights, ord=2)
    return val

def getDegreeMatrix(G):
    degs = [val/2 for (node, val) in G.degree()]
    return np.diag(degs)

def getNormalizedAdjacencyMatrix(G):
    adj = nx.to_numpy_array(G)
    deg = getDegreeMatrix(G)
    # middle
    mid = adj + np.eye(len(adj))

    # calculate deg -1/2
    deg_diags = np.diag(deg)
    inv_sqrt = np.power(deg_diags, -1/2)
    deg_root = np.diag(inv_sqrt)

    return np.matmul(deg_root, np.matmul(mid, deg_root))

def get_lipschitz_bound(G, Gp, model):
    layers = getLayers(model)
    d = G.number_of_nodes()
    prod_weights = getProdOfLayerWeights(model, layers)
    
    norm_adj_G = getNormalizedAdjacencyMatrix(G)
    norm_adj_Gp = getNormalizedAdjacencyMatrix(Gp)
    
    sparse_norm_adj_G = scp.sparse.csr_matrix(norm_adj_G)
    sparse_norm_adj_Gp = scp.sparse.csr_matrix(norm_adj_Gp)
    
    E = sparse_norm_adj_G - sparse_norm_adj_Gp
    
    E_norm = scp.sparse.linalg.norm(E, ord=2)
    
    return layers * np.sqrt(d) * E_norm * prod_weights

def random_perturbation(ptb_rate, G):
    Gp = G.copy()

    budget = math.floor(ptb_rate * G.number_of_edges() * 1/2)

    i = 0

    while i < budget:
        u = random.randint(0, Gp.number_of_nodes() - 1)
        v = random.randint(0, Gp.number_of_nodes() - 1)

        if u != v and not Gp.has_edge(u, v):
            Gp.add_edge(u, v)
            Gp.add_edge(v, u)
            i += 1
        
    return Gp

def get_simplified_lipschitz(model):
    layers = getLayers(model)
    prod_weights = getProdOfLayerWeights(model, layers)
    
    return layers * prod_weights