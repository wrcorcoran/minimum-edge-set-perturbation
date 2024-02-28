# Findings Regarding Homophily
### Ideas:
- Are there any characteristics of a homophilic *class*? Like, can I add more edges to a highly homophilic class or is it generally impacted by the homophily of the nodes?
    - [Characterizing Graph Datasets for Node Classification: Homophily–Heterophily Dichotomy and Beyond](https://arxiv.org/pdf/2209.06177.pdf)
    - [IS HOMOPHILY A NECESSITY FOR GRAPH NEURAL NETWORKS?](https://arxiv.org/pdf/2106.06134.pdf)

### Week 6
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

##### Very interesting Results! [MISTAKENLY DROPPED HOMOPHILIC ASPECT]
- I did an entire experiment before realizing that I'd dropped the homophilic aspect of the experiment. I will pass these results to Niyati for her findings.
- Will add basic results to research log.

#### Possible Ideas:
- Is there some pattern which would not change the value? Some geometric pattern?
- Can we increase the number of similar nodes by some threshold?
  - Yes! See "Interesting Results"
- Can we selectively pair nodes with higher (or lower) homophilic degrees? What threshold would emerge to cause no change?

#### Overarching questions:
- Is there an intelligent way we could pick edges such that the there would be no change?
- To some degree, the change seems to relate most to the degree of nodes (no pun intended).

### Week 8
Here are my findings so far in Week 8:
    - **What happens if we turn a connected component, limited to a certain class, into a clique?**
      - Due to the small size of the CORA dataset, there are few connected components with size $> 1$ that reside in a single class. $2$ to be exact. While the accuracy was not changed, only $3$ edges were added, leaving this experiment relatively uninsightful.
      Implementation:
      ```
        data = dataset.get_data()[0]
        modified_graph = data
        \
        init_edges = len(modified_graph.edge_index[1])
        \
        G, x, y, train_mask, test_mask = convert_to_networkx(modified_graph)
        \
        cc = sorted(nx.strongly_connected_components(G), key=len, reverse=True)
        \
        for i in cc:
            s = set(i)
            for j in range(0, num_classes):
                if s == s.intersection(homophilic_set[j]):
                    make_clique(G, s)
        \
        modified_graph = convert_to_pyg(G, x, y, train_mask, test_mask)
        final_edges = len(modified_graph.edge_index[1])
        \
        output_accuracy_change(ground_truth, test_model(model, modified_graph)) 
        number_added_edges(init_edges, final_edges, is_undirected=True)
      ```
    - **What happens if we turn a connected component, limited to a certain class, with a certain density into a clique?**
      - Similar to the above example, the CORA dataset is too limited. No nodes were changed.
      Implementation:
      ```
        data = dataset.get_data()[0]
        modified_graph = data
        \
        init_edges = len(modified_graph.edge_index[1])
        \
        G, x, y, train_mask, test_mask = convert_to_networkx(modified_graph)
        \
        cc = sorted(nx.strongly_connected_components(G), key=len, reverse=True)
        \
        for i in cc:
            s = set(i)
            for j in range(0, num_classes):
                if (s == s.intersection(homophilic_set[j])) and (nx.density(G.subgraph(list(s))) > 0.2):
                    make_clique(G, s)
        \
        modified_graph = convert_to_pyg(G, x, y, train_mask, test_mask)
        final_edges = len(modified_graph.edge_index[1])
        \
        output_accuracy_change(ground_truth, test_model(model, modified_graph)) 
        number_added_edges(init_edges, final_edges, is_undirected=True)
      ```
    - **What happens if we turn a connected component, when considering ONLY edges in a certain class, into a clique?**
      - This is where things start to get interesting.
      - First, I made a subgraph specific to the class, and then ran a connected components algorithm on it.
      - Upon cliquing an entire connected component, $4211$ edges are added, an increase of $79.78\%$.
      - The accuracy *rose* by 0.0230. 
      - While not stagnant, this is an incredibly minimal change for increasing the edge set by $4211$ edges.
      Implementation:
        ```
        data = dataset.get_data()[0]
        modified_graph = data
        \
        init_edges = len(modified_graph.edge_index[1])
        \
        G, x, y, train_mask, test_mask = convert_to_networkx(modified_graph)
        \
        for j in range(0, len(homophilic_set)):
            new_G = G.subgraph(homophilic_set[j])
            cc = sorted(nx.strongly_connected_components(new_G), key=len, reverse=True)
            for i in cc:
                s = set(i)
                if len(s) > 1:
                    make_clique(G, s)
        \
        modified_graph = convert_to_pyg(G, x, y, train_mask, test_mask)
        final_edges = len(modified_graph.edge_index[1])
        \
        output_accuracy_change(ground_truth, test_model(model, modified_graph)) 
        number_added_edges(init_edges, final_edges, is_undirected=True)
        ```
    - **What happens if we turn a connected component with a certain density *c*, when considering ONLY edges in a certain class, into a clique?**
      - Again, I made subgraphs respective to the class.
      - For density threshold $c$, I tested all multiples of $0.05$ from $0.05$ to $0.75$. The results are as follows:
            | Value of C | Change in Edges | Change in Accuracy |
            |:------------:|:--------:|:---------:|
            | 0.05  |  2076 (39.33%) |   +0.0090  |
            | 0.1     |   816 (15.46%) |    +0.0080   |
            | 0.15  |  414 (7.84%) |   +0.0050  |
            | 0.2     |   338 (6.4%)   |    +0.0040   |
            | 0.25  |  161 (3.05%) |   +0.0020  |
            | 0.3     |   140 (3.05%)  |    +0.0020   |
            | 0.35  |  120 (2.65%) |   +0.0020  |
            | 0.4     |   120 (2.27%)  |    +0.0020   |
            | 0.45  |  99 (1.88%) |   +0.0010  |
            | 0.5     |   83 (1.57%)  |    +0.0010   |
            | 0.55  |  20 (0.38%) |   +0.0010  |
            | 0.6     |   20  (0.38%) |    +0.0010   |
            | 0.65  |  20  (0.38%) |   +0.0010  |
            | 0.7     |   0   |    0   |
            | 0.75  |  0  |   0  |
      - We can see you are able to add an extremely large number of edges without much change in classification. However, there are no instances where exactly $0.00\%$ accuracy is changed.
      Implementation:
      ```
        c_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.80, 0.85]
        \
        for c in c_values:
        data = dataset.get_data()[0]
        modified_graph = data
        \
        init_edges = len(modified_graph.edge_index[1])
        \
        G, x, y, train_mask, test_mask = convert_to_networkx(modified_graph)
        \
        for j in range(0, len(homophilic_set)):
            new_G = G.subgraph(homophilic_set[j])
            cc = sorted(nx.strongly_connected_components(new_G), key=len, reverse=True)
            for i in cc:
                s = set(i)
                if len(s) > 1 and nx.density(G.subgraph(i)) >= c:
                    make_clique(G, s)
        \
        modified_graph = convert_to_pyg(G, x, y, train_mask, test_mask)
        final_edges = len(modified_graph.edge_index[1])
        \
        output_accuracy_change(ground_truth, test_model(model, modified_graph)) 
        number_added_edges(init_edges, final_edges, is_undirected=True)
        print("For c value:", c)
      ```
    - **What happens if we increase the density of a connected component, when considering ONLY edges in a certain class, by a certain threshold?**
      - For the density increase constant, $c$, I tested: $1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55,$$1.6, 1.65, 1.70, 1.75, 1.80, 1.85, 1.9, 1.95, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100$
      - The results are as follows:
          | Value of C | Change in Edges | Change in Accuracy |
          |:------------:|:-------------------:|:-------------------:|
          | 1.05           | 62 (1.17%)               | +0.0010                   |
          | 1.1             | 78 (1.48%)               | +0.0020                   |
          | 1.15           | 99 (1.88%)               | +0.0050                   |
          | 1.2             | 114 (2.16%)             | +0.0040                   |
          | 1.25           | 135 (2.56%)             | +0.0030                   |
          | 1.3             | 159 (3.01%)             | +0.0040                   |
          | 1.35           | 186 (3.52%)             | +0.0100                   |
          | 1.4             | 202 (3.83%)             | +0.0080                   |
          | 1.45           | 227 (4.30%)             | +0.0110                   |
          | 1.5             | 243 (4.60%)             | +0.0100                   |
          | 1.55           | 264 (5.00%)             | +0.0100                   |
          | 1.6             | 281 (5.32%)             | +0.0070                   |
          | 1.65           | 305 (5.78%)             | +0.0140                   |
          | 1.7             | 330 (6.25%)             | +0.0110                   |
          | 1.75           | 350 (6.63%)             | +0.0120                   |
          | 1.8             | 367 (6.95%)             | +0.0160                   |
          | 1.85           | 390 (7.39%)             | +0.0130                   |
          | 1.9             | 408 (7.73%)             | +0.0160                   |
          | 1.95           | 424 (8.03%)             | +0.0150                   |
          | 2               | 433 (8.20%)             | +0.0140                   |
          | 3               | 767 (14.53%)           | +0.0200                   |
          | 4               | 1076 (20.39%)         | +0.0200                   |
          | 5               | 1363 (25.82%)         | +0.0220                   |
          | 6               | 1620 (30.69%)         | +0.0240                   |
          | 7               | 1862 (35.28%)         | +0.0240                   |
          | 8               | 2086 (39.52%)         | +0.0230                   |
          | 9               | 2281 (43.22%)         | +0.0230                   |
          | 10             | 2464 (46.68%)         | +0.0230                   |
          | 100           | 4211 (79.78%)         | +0.0230                   |
      -  Again, we are able to add a large amount of nodes with *little change*, but there are no changes which results in $0$ change.
      -  However, the most interesting result seems to be that there is a convergence, as to the amount the model changes. I have **zero** explanation as to why, and it's something I'm going to be thinking about/looking into over the next few weeks.
      Implementation:
      ```
        c_values = [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.70, 1.75, 1.80, 1.85, 1.9, 1.95, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]
        # strange...it starts to plateau after 5-6
        # definitely something, I'll have to think on what...
        \
        def increase_density(G, s, threshold):
            while nx.density(G.subgraph(s)) < min(threshold, 1):
                random_pair = random.sample(s, 2)
                add_edge(G, random_pair[0], random_pair[1], undirected=True)
        \
        for c in c_values:
            data = dataset.get_data()[0]
            modified_graph = data
         \   
            init_edges = len(modified_graph.edge_index[1])
          \  
            G, x, y, train_mask, test_mask = convert_to_networkx(modified_graph)
        \
            for j in range(0, len(homophilic_set)):
                new_G = G.subgraph(homophilic_set[j])
                cc = sorted(nx.strongly_connected_components(new_G), key=len, reverse=True)
                for i in cc:
                    s = set(i)
                    if len(s) > 1:
                        threshold = nx.density(G.subgraph(s))
                        increase_density(G, s, c*threshold)
             \   
            modified_graph = convert_to_pyg(G, x, y, train_mask, test_mask)
            final_edges = len(modified_graph.edge_index[1])
            \
            output_accuracy_change(ground_truth, test_model(model, modified_graph)) 
            number_added_edges(init_edges, final_edges, is_undirected=True)
            print("For c value:", c)
      ```
   - **What happens if we clique the graph irrespective to classes? Meaning, all connected components become cliques?**
     - The model fails. Expectedly.
     - Over $3$ million edges were added and accuracy decreased by $-0.5520$.
     - This is expected. Moreso, I was interested to see what would happen.
     Implementation:
     ```
     data = dataset.get_data()[0]
     \
        modified_graph = data
      \  
        init_edges = len(modified_graph.edge_index[1])
       \ 
        G, x, y, train_mask, test_mask = convert_to_networkx(modified_graph)
        \
        cc = sorted(nx.strongly_connected_components(G), key=len, reverse=True)
        \
        for i in cc:
            s = set(i)
            make_clique(G, s)
        \
        modified_graph = convert_to_pyg(G, x, y, train_mask, test_mask)
        final_edges = len(modified_graph.edge_index[1])
        \
        output_accuracy_change(ground_truth, test_model(model, modified_graph)) 
        number_added_edges(init_edges, final_edges, is_undirected=True)
     ```