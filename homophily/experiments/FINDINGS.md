# Findings Regarding Homophily

### February 11th
##### What happens if all edges in a class are connected to a single node?
- On the CORA dataset, there was a change $0.0020$ from the ground truth.
- How was this implemented?
    ``` Python
    for i in range(0, num_classes):
    for j in range(1, len(homophilic_set[i])):
        if not modified_graph.has_edges_between([homophilic_set[i][0]], j):
            modified_graph.add_edges(j, homophilic_set[i][0])
            modified_graph.add_edges(homophilic_set[i][0], j)
    ```
- The first node in each class was selected and all other nodes were connected.
- How could we modify this?
  - Are there specific nodes which would result in no change (those with higher degree? those with lower degree?)?
  - What happens if all nodes are connected to each other? This is relatively computationally expensive, but can be done on a smaller graph like ```CORA```. 

##### What happens if only two edges in a single class are connected?
- On the CORA dataset, there was a change anyway from $0.0$ to $0.0030$ respective to the ground truth.
- How was this implemented?
    ``` Python
    for i in range(0, num_classes):
        r_node = random.choice(homophilic_set[i])
        n_node = random.choice(homophilic_set[i])

        while r_node == n_node or modified_graph.has_edges_between(n_node, r_node):
            n_node = random.choice(homophilic_set[i])

        modified_graph.add_edges(r_node, n_node)
        modified_graph.add_edges(n_node, r_node)
        print("added: ", r_node, " - ", n_node)
    ```
- Two random nodes (in the same class) were connected if they weren't previously.
- How could we modify this?
  - Is there a number of random edges we can add? At what threshold does the change begin to grow past $0.0030$?

