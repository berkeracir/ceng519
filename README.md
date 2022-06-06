# CENG519 Network Security - Term Project

The dockerfile is for generating a container that establishes the development environment for CENG519 which includes the required packages for Microsoft SEAL and EVA. You will implement a graph algorithm preserving the privacy of the graph. Note that, CKKS used by EVA is not powerful enough to do all kinds of computations. There is no comparison for instance. 

## Algorithm: [Find the Number of Islands](https://www.geeksforgeeks.org/find-number-of-islands/)

> Given a boolean 2D matrix, find the number of islands. A group of connected 1s forms an island and a cell in the 2D matrix can be connected to 8 neighbours. This is a variation of the standard problem: “Counting the number of connected components in an undirected graph”. A connected component of an undirected graph is a subgraph in which every two vertices are connected to each other by a path(s), and which is connected to no other vertices outside of the subgraph.
> 
> The problem can be easily solved by applying DFS (Depth First Search) on each component. In each DFS call, a component or a sub-graph is visited. Then, DFS is called on the next un-visited component. The number of calls to DFS gives the number of connected components (or islands in this problem). BFS (Breadth First Search) can also be used.
> 
> **Time complexity: O(ROW x COL)**

For example, the 2D matrix below has 5 islands:

![Example 2D matrix with 5 islands](/figures/input_matrix.png "Input Matrix")

## Running the algorithm

```
# build the docker image
docker build --progress=plain -t berkeracir/ceng519-fhe .
# run the docker image
docker run -v $(pwd)/project:/development/project -it berkeracir/ceng519-fhe /bin/bash

# run the parallel version of the implemented algorithm
python3 project/fhe_parallel.py
# run the parallel and looped version of the implemented algorithm
python3 project/fhe_parallel_loop.py
# run the vectorized version of the implemented algorithm
python3 project/fhe_vectorized.py
# run the implemented algorithm without using FHE libraries and compilers
python3 project/without_fhe.py
```