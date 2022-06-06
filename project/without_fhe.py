from math import ceil
import numpy as np

# directions: left, right, up, down
# {direction: (shift amount, axis, index of zeroed col/row)}
SHIFT_DICT = {'l': (-1,1,-1), 'r': (1,1,0), 'u': (-1,0,-1), 'd': (1,0,0)}

# Shift the given numpy array along the specified direction
# Note that shift operation is not circular (e.g. [1,2,3] << 1 == [2,3,0]).
def shift(matrix, direction):
	shift, axis, remove = SHIFT_DICT[direction]
	result = np.copy(matrix)
	result = np.roll(result, shift=shift, axis=axis)

	if direction in ['l', 'r']:
		result[:,remove] = 0
	else:
		result[remove,:] = 0

	return result

# Convert input matrix into directed adjacency matrix
# For example, input matrix of shape (3,3):
# arr = [[a, b, c],
#        [d, e, f],
#        [g, h, i]]
# will be converted into the following adjacency matrix of shape (9,9):
# adj = [[0, b, 0, d, e, 0, 0, 0, 0],
#        [a, 0, c, d, e, f, 0, 0, 0],
#        [0, b, 0, 0, e, f, 0, 0, 0],
#        [a, b, 0, 0, e, 0, g, h, 0],
#        [a, b, c, d, 0, f, g, h, i],
#        [0, b, c, 0, e, 0, 0, h, i],
#        [0, 0, 0, d, e, 0, 0, h, 0],
#        [0, 0, 0, d, e, f, g, 0, i],
#        [0, 0, 0, 0, 0, e, f, h, 0]]
# Rows/Columns of adjacency matrix represents a, b, c, d, e, f, g, h and i respectively.
# If adj[i,j] is equal to 0, then node from arr with row i//3 and column i%3 is not adjacent to node from arr with row j//3 and column j%3.
# Otherwise, node from arr with row i//3 and column i%3 is adjacenct to node from arr with row j//3 and column j%3.
def adjacencyMatrix(matrix, n):
	N = n * n
	result = np.zeros(shape=(N,N))

	for i in range(n):
		for j in range(n):
			col = n * i + j

			for x in [-1, 0, 1]:
				for y in [-1, 0, 1]:
					if x == 0 and y == 0:
						continue
					if 0 <= i+x < n and 0 <= j+y < n:
						row = n * (i + x) + (j + y)
						result[row, col] = matrix[i, j]

	return result

# Reduce the number of ones in the input matrix
def reduceOnes(matrix):
	result = np.copy(matrix)

	r_shifted = shift(result, 'r')
	dr_shifted = shift(r_shifted, 'd')
	d_shifted = shift(result, 'd')
	dl_shifted = shift(d_shifted, 'l')

	result = result * (result + r_shifted + d_shifted + dr_shifted + dl_shifted)
	return result

# Compute distances between each elements
def computeDistances(matrix, n):
	N = n * n
	adj_matrix = adjacencyMatrix(matrix, n)
	result = np.copy(adj_matrix)

	for row in range(N):
		for repeat in range(ceil(N/2)):
			for col in range(N):
				if row == col:
					continue
				result[row] += result[row, col] * adj_matrix[col]

	return result 

# Compute distances between each vectorized elements
def computeDistances_vectorized(matrix, n):
	N = n * n
	adj_matrix = adjacencyMatrix(matrix, n).ravel()
	result = np.copy(adj_matrix)

	for repeat in range(ceil(N/2)):
		for shift in range(N):
			adjacency = np.tile(adj_matrix[N*shift:N*shift+N], N)
			extended = np.zeros(shape=(N**2))
			for i in range(N):
				extended[i*N:i*N+N] = np.tile(result[i*N+shift], N)
			result += extended * adjacency
	
	return result.reshape((N,N))

# Count the islands
def countIslands(matrix, distances, n):
	N = n * n
	indices = {}
	islands = set()
	for row in range(N):
		if matrix[row//n, row%n] == 1:
			indices[(row//n, row%n)] = [(x//n, x%n) for x in np.argwhere(distances[row] != 0).ravel() if x != row and matrix[x//n, x%n] == 1]

			if not (set(indices[row//n, row%n]) & islands):
				islands.add((row//n, row%n))

	return sorted(islands), len(islands)


if __name__ == "__main__":
	n = int(input("Input Size: "))
	input_matrix = np.random.randint(2, size=(n,n))

	reduced_input = reduceOnes(input_matrix)
	distances = computeDistances(input_matrix, n)
	distances_vectorized = computeDistances_vectorized(input_matrix, n)
	islands, count = countIslands(reduced_input, distances, n)
	islands_vectorized, count_vectorized = countIslands(reduced_input, distances, n)
	distances[distances != 0] = 1

	print("\nInput Matrix:")
	print(input_matrix, '\n')
	print("Reduced Input Matrix:")
	print(reduced_input, '\n')
	print(f"Islands: {islands} {'==' if islands == islands_vectorized else '!='} {islands_vectorized} (Vectorized)")
	print(f"Island Count: {count} {'==' if count == count_vectorized else '!='} {count_vectorized} (Vectorized)")