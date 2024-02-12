# Findings Regarding Homophily

### Week 5
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
- On the CORA dataset, there was a change anywhere from $0.0$ to $0.0030$ respective to the ground truth.
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

##### What happens if all edges in a single class are connected?
- As expected, this took a little while to calculate. There are roughly 1000 nodes in the ```test``` mask of the CORA dataset. Therefore, in the worst case there was $1\,000\,000$ edge additions.
- How was this implemented?
    ```Python
    for i in range(0, num_classes):
        for j in range(0, len(homophilic_set[i])):
            for k in range(j + 1, len(homophilic_set[i])):
                j_node = homophilic_set[i][j]
                k_node = homophilic_set[i][k]
                if j_node == k_node or modified_graph.has_edges_between(j_node, k_node):
                    continue

                modified_graph.add_edges(j_node, k_node)
                modified_graph.add_edges(k_node, j_node)
                print("added: ", j_node, " - ", k_node)
    ```
- It may come to no surprise, but there was a *large* change in results. The results changed by $-0.2310$. Evidently, this would not be a valid minimum set. It wasn't expected to be, rather, it was just a test for the runtime of these processes, but was interesting to look at.

##### Interesting Results! What happens when edges are added relative to the number of homophilic edges a node already has?
I developed this according to the formula
$$|added(n)| = \lfloor c \times h(n) \rfloor$$
where $c$ is some constant and $h(n)$ represents the number of homophilic edges. $added(n)$ is the set of edges added from a specific node. Basically, this limits the number of edges which can be added according to the pre-existing degree of homophilic edges. 
For example, given $c = 0.1$, if a node $u$ has $23$ homophilic edges then $2$ randomly-selected homophilic edges $v_1, v_2$ will be added from this node, $(u, v_1)$ and $(u, v_2)$. 
- Initially, I tried using ceiling (this fails for obvious reasons).
- The implementation is as follows:
    ```Python
    for i in range(0, num_classes):
        for j in range(0, len(homophilic_set[i])):
            edges = modified_graph.out_edges(homophilic_set[i][j])

            same_class_edges = set()
            for e in edges[1]:
                if (e.item() in homophilic_set[i]):
                    same_class_edges.add(e.item())

            for k in range(0, math.floor(len(same_class_edges) * c)):
                a_node = homophilic_set[i][j]
                b_node = random.choice(homophilic_set[i])

                while a_node == b_node or modified_graph.has_edges_between(a_node, b_node):
                    b_node = random.choice(homophilic_set[i])

                modified_graph.add_edges(a_node, b_node)
                modified_graph.add_edges(b_node, a_node)
                class_and_added[i].append(len(same_class_edges))
    ```
- Here are the follow results:
  -  $c = 0.10$, $4$ added edges, **no change in accuracy**
  -  $c = 0.15$, $11$ added edges, **accuracy change $= 0.0010$** 
  -  $c = 0.20$, $28$ added edges, **accuracy change $= -0.0040$**
  -  $c = 0.25$, $47$ added edges, **accuracy change $= -0.0030$**
  -  $c = 0.30$, $55$ added edges, **accuracy change $= -0.0070$**
  -  $c = 1/3$, $128$ added edges, **accuracy change $= -0.0100$**
  -  $c = 0.35$, $138$ added edges, **accuracy change $= -0.0250$**
  -  $c = 0.50$, $419$ added edges, **accuracy change $= -0.0460$**   
-  Assessment:
   -  I feel it was expected that the more edges you added, the more your accuracy is going to change (likely decrease).
   -  However, it's notable that you can increase by homophilic degree of a node $1.25$ times without seeing a large change. 
      -  It seems possible there's an intelligent choice of edges (rather than random!), which would increase the capabilities of this threshold and maintain accuracy.
- Interesting questions this poses:
  - Is this a trend that extends to degree-only heuristics? Obviously, this is a combination of degree analysis and homophily, but it trends towards utilizing the definition of homophily. However, if it's a trend regardless, then, it leaves homophily non-important.
  - Can we selectively pair nodes with higher (or lower) homophilic degrees? What threshold would emerge to cause no change?
  - This seems to show a general trend of edges added increasing change in accuracy. However, it's possible that this isn't always true. 

#### Possible Ideas:
- Is there some pattern which would not change the value? Some geometric pattern?
- Can we increase the number of similar nodes by some threshold?
  - Yes! See "Interesting Results"
- Can we selectively pair nodes with higher (or lower) homophilic degrees? What threshold would emerge to cause no change?

#### Overarching questions:
- Is there an intelligent way we could pick edges such that the there would be no change?
- To some degree, the change seems to relate most to the degree of nodes (no pun intended).