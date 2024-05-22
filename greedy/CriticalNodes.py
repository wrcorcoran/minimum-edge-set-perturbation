import networkx as nx
import torch
import numpy as np

class CriticalNode:
    def __init__(self, node, homophily_values, degrees, labels, two_hop_neighborhood_sizes):
        self.node = node
        self.homophily = homophily_values[node]
        self.degree = degrees[node]
        self.ground_truth = labels[node].item()
        self.two_hop_neighborhood_size = two_hop_neighborhood_sizes[node]

    def print_data(self):
        print(f"\tNode: {self.node}")
        print(f"\t\tHomophily: {self.homophily}")
        print(f"\t\tDegree: {self.degree}")
        print(f"\t\tGround Truth: {self.ground_truth}")
        print(f"\t\tTwo-hop Neighborhood Size: {self.two_hop_neighborhood_size}")

class CriticalEdge:
    def __init__(self, node1, node2, homophily_values, degrees, labels, two_hop_neighborhood_sizes):
        self.node1 = CriticalNode(node1, homophily_values, degrees, labels, two_hop_neighborhood_sizes)
        self.node2 = CriticalNode(node2, homophily_values, degrees, labels, two_hop_neighborhood_sizes)

    def print_data(self):
        print("Edge between nodes:")
        self.node1.print_data()
        self.node2.print_data()

class CriticalEdges:
    def __init__(self, edges):
        self.edges = edges

    def calculate_statistics(self):
        degrees = [(edge.node1.degree + edge.node2.degree) / 2 for edge in self.edges]
        homophily_values = [(edge.node1.homophily + edge.node2.homophily) / 2 for edge in self.edges]
        two_hop_neighborhood_sizes = [(edge.node1.two_hop_neighborhood_size + edge.node2.two_hop_neighborhood_size) / 2 for edge in self.edges]
        ground_truth_values = [edge.node1.ground_truth for edge in self.edges]

        avg_degree = np.mean(degrees)
        min_degree = np.min(degrees)
        max_degree = np.max(degrees)

        min_homophily = np.min(homophily_values)
        max_homophily = np.max(homophily_values)
        avg_homophily = np.mean(homophily_values)

        min_two_hop = np.min(two_hop_neighborhood_sizes)
        max_two_hop = np.max(two_hop_neighborhood_sizes)
        avg_two_hop = np.mean(two_hop_neighborhood_sizes)

        ground_truth_count = torch.tensor(ground_truth_values).bincount().tolist()

        return {
            "avg_degree": avg_degree,
            "min_degree": min_degree,
            "max_degree": max_degree,
            "min_homophily": min_homophily,
            "max_homophily": max_homophily,
            "avg_homophily": avg_homophily,
            "min_two_hop": min_two_hop,
            "max_two_hop": max_two_hop,
            "avg_two_hop": avg_two_hop,
            "ground_truth_count": ground_truth_count
        }

    def print_statistics(self):
        stats = self.calculate_statistics()
        print("----")
        print(f"Average Degree: {stats['avg_degree']}")
        print(f"Min Degree: {stats['min_degree']}")
        print(f"Max Degree: {stats['max_degree']}")
        print(f"Min Homophily: {stats['min_homophily']}")
        print(f"Max Homophily: {stats['max_homophily']}")
        print(f"Average Homophily: {stats['avg_homophily']}")
        print(f"Min Two-hop Neighborhood Size: {stats['min_two_hop']}")
        print(f"Max Two-hop Neighborhood Size: {stats['max_two_hop']}")
        print(f"Average Two-hop Neighborhood Size: {stats['avg_two_hop']}")
        print(f"Ground Truth Classification Counts: {stats['ground_truth_count']}")
        print("----")

        for edge in self.edges:
            edge.print_data()
            print()
