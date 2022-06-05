import timeit
from math import ceil
import numpy as np
from random import randint
import os

from eva import EvaProgram, Input, Output, evaluate, Expr
from eva.ckks import CKKSCompiler
from eva.seal import generate_keys
from eva.metric import valuation_mse


VERBOSE = False

# directions: left, right, up, down
# direction: (shift amount, axis, index of zeroed col/row)
SHIFT_DICT = {'l': (-1,1,-1), 'r': (1,1,0), 'u': (-1,0,-1), 'd': (1,0,0)}

# Shift the given numpy array along the specified direction.
# Note that shift operation is not circular (e.g. [1,2,3] << 1 == [2,3,0]).
def shift(arr, direction):
	shift, axis, remove = SHIFT_DICT[direction]
	result = np.copy(arr)
	result = np.roll(result, shift=shift, axis=axis)

	if direction in ['l', 'r']:
		result[:,remove] = 0
	else:
		result[remove,:] = 0

	return result

# Convert input matrix into directed adjacency matrix. For example, input matrix of shape (3,3):
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
# If adj[i,j] is equal to 0, then node from arr with row i//3 and column i%3 is not neighbor of node from arr with row j//3 and column j%3.
# Otherwise, node from arr with row i//3 and column i%3 is neighbor of node from arr with row j//3 and column j%3.
def adjacencyMatrix(arr, n, zero_expr):
	N = n * n	# matrix size is squared
	result = np.zeros(shape=(N,N), dtype=Expr)

	for i in range(n):
		for j in range(n):
			col = n * i + j

			for x in [-1, 0, 1]:
				for y in [-1, 0, 1]:
					row = n * (i + x) + (j + y)
					if x == 0 and y == 0:
						result[row, col] = zero_expr	# https://github.com/microsoft/SEAL/issues/200
					elif 0 <= i+x < n and 0 <= j+y < n:
						result[row, col] = arr[i, j]

	return result

def printReduceOnes(inputs, outputs, reference, n, random_index, timings, verbose=False):
	if verbose:
		print(f"\nREDUCE ONES:\n" +
				f"-Compile Time: {timings['compile']}\n" +
				f"-Key Generation Time: {timings['keyGeneration']}\n" +
				f"-Encryption Time: {timings['encryption']}\n" + 
				f"-Execution Time: {timings['execution']}\n" +
				f"-Decryption Time: {timings['decryption']}\n" +
				f"-Reference Execution Time: {timings['referenceExecution']}\n" +
				f"-MSE: {timings['mse']}")

		input_matrix = np.zeros((n,n))
		output_matrix = np.zeros((n,n))
		reference_matrix = np.zeros((n,n))

		for i in range(n):
			for j in range(n):
				key = f"n_{i}_{j}"
				input_matrix[i,j] = inputs[key][random_index]
				output_matrix[i,j] = outputs[key][random_index]
				reference_matrix[i,j] = reference[key][random_index]
	
		input_matrix = np.rint(input_matrix).astype(int)
		output_matrix = np.rint(output_matrix).astype(int)
		reference_matrix = np.rint(reference_matrix).astype(int)
		print("Input Matrix:")
		print(input_matrix)
		print("Reduced Matrix:")
		print(output_matrix)
		print(f"Reduced Matrix {'==' if (np.array_equal(output_matrix, reference_matrix)) else '!='} Reference")

def printComputeDistances(inputs, outputs, reference, n, random_index, timings, verbose=False):
	if verbose:
		print(f"\nCOMPUTE DISTANCES:\n" +
				f"-Compile Time: {timings['compile']}\n" +
				f"-Key Generation Time: {timings['keyGeneration']}\n" +
				f"-Encryption Time: {timings['encryption']}\n" + 
				f"-Execution Time: {timings['execution']}\n" +
				f"-Decryption Time: {timings['decryption']}\n" +
				f"-Reference Execution Time: {timings['referenceExecution']}\n" +
				f"-MSE: {timings['mse']}")

		N = n * n	# matrix size is squared
		input_matrix = np.zeros((n,n))
		output_matrix = np.zeros((N,N))
		reference_matrix = np.zeros((N,N))

		for i in range(N):
			for j in range(N):
				key = f"n_{i}_{j}"
				if i < n and j < n:
					input_matrix[i,j] = inputs[key][random_index]
				output_matrix[i,j] = outputs[key][random_index]
				reference_matrix[i,j] = reference[key][random_index]
	
		input_matrix = np.rint(input_matrix).astype(int)
		output_matrix = np.rint(output_matrix).astype(int)
		reference_matrix = np.rint(reference_matrix).astype(int)
		print("Input Matrix:")
		print(input_matrix)
		print("Distance Matrix:")
		print(output_matrix)
		print(f"Distance Matrix {'==' if (np.array_equal(output_matrix, reference_matrix)) else '!='} Reference")

# Prepare input matrices where each matrix has shape of (n,n) and there are `vec_size` matrices.
# Returns dictionary of matrices: {n_i_j: [x,y,z...]} (i.e. the (i,j)th element of the first matrix is 1).
def prepareInputs(n, vec_size):
	inputs = {}
	
	for vec in range(vec_size):
		# Create a matrix of size (n,n) with random 0s and 1s.
		matrix = np.random.randint(2, size=(n,n))

		for i in range(n):
			for j in range(n):
				key = f"n_{i}_{j}"
				if key not in inputs.keys():
					inputs[key] = [matrix[i,j]]
				else:
					inputs[key].append(matrix[i,j])

	return inputs

# Analytic service that reduces the number of ones.
def reduceOnesAnalytics(input, n):
	# Create a numpy zeros matrix with Expr data type for input data
	matrix = np.zeros((n,n), dtype=Expr)
	for i in range(n):
		for j in range(n):
			key = f"n_{i}_{j}"
			# From the input data create the matrix
			matrix[i,j] = input[key]
	
	r_shifted = shift(matrix, 'r')		# right shifted matrix
	rd_shifted = shift(r_shifted, 'd')	# right and down shifted matrix
	d_shifted = shift(matrix, 'd')		# left shifted matrix
	ld_shifted = shift(d_shifted, 'l')	# left and down shifted matrix

	# Multiply the matrix with the summation of itself and its (r, rd, d, ld) shifted versions
	# This reduces the number of 1's in the matrix.
	matrix = matrix * (matrix + r_shifted + rd_shifted + d_shifted + ld_shifted)

	return matrix

# EVA Program that reduces the number of ones.
def reduceOnes(inputs, n, vec_size, config, random_index):
	reduceOnesProgram = EvaProgram("Reduce ones", vec_size=vec_size)
	with reduceOnesProgram:
		input = {}
		for i in range(n):
			for j in range(n):
				key = f"n_{i}_{j}"
				input[key] = Input(key)
		
		output_matrix = reduceOnesAnalytics(input, n)
		for i in range(n):
			for j in range(n):
				key = f"n_{i}_{j}"
				Output(key, output_matrix[i,j])
	
	prog = reduceOnesProgram
	prog.set_output_ranges(30)
	prog.set_input_scales(30)

	# Compilation
	start = timeit.default_timer()
	compiler = CKKSCompiler(config=config)
	compiled_multfunc, params, signature = compiler.compile(prog)
	compileTime = (timeit.default_timer() - start) * 1000.0 #ms

	# Key Generation
	start = timeit.default_timer()
	public_ctx, secret_ctx = generate_keys(params)
	keyGenerationTime = (timeit.default_timer() - start) * 1000.0 #ms
	
	# Encryption
	start = timeit.default_timer()
	encInputs = public_ctx.encrypt(inputs, signature)
	encryptionTime = (timeit.default_timer() - start) * 1000.0 #ms

	# Execution
	start = timeit.default_timer()
	encOutputs = public_ctx.execute(compiled_multfunc, encInputs)
	executionTime = (timeit.default_timer() - start) * 1000.0 #ms

	# Decryption
	start = timeit.default_timer()
	outputs = secret_ctx.decrypt(encOutputs, signature)
	decryptionTime = (timeit.default_timer() - start) * 1000.0 #ms

	# Evaluation
	start = timeit.default_timer()
	reference = evaluate(compiled_multfunc, inputs)
	referenceExecutionTime = (timeit.default_timer() - start) * 1000.0 #ms

	# Approximation
	mse = valuation_mse(outputs, reference) # since CKKS does approximate computations, this is an important measure that depicts the amount of error

	timings = {'compile': compileTime,
			   'keyGeneration': keyGenerationTime,
			   'encryption': encryptionTime,
			   'execution': executionTime,
			   'decryption': decryptionTime,
			   'referenceExecution': referenceExecutionTime,
			   'mse': mse
			  }
	printReduceOnes(inputs, outputs, reference, n, random_index, timings, verbose=VERBOSE)
	return outputs, timings

# Analytic service that computes distances between each elements.
def computeDistancesAnalytics(input, n):
	# Create a numpy zeros matrix with Expr data type for input data
	matrix = np.zeros(shape=(n,n), dtype=Expr)
	for i in range(n):
		for j in range(n):
			key = f"n_{i}_{j}"
			# From the input data create the matrix
			matrix[i,j] = input[key]
	
	N = n * n	# adjacency matrix size is square of input's size
	# Get the adjacency matrix of input matrix 
	adj_matrix = adjacencyMatrix(matrix, n, input['zero'])
	distances = np.copy(adj_matrix)

	for row in range(N):
		for _ in range(ceil(N/2)):
			for col in range(N):
				if row == col:	# ignore self-loops
					continue
				# distances[row] += np.full(shape=(N,), fill_value=(matrix[row//n, row%n] * distances[row, col])) * adj_matrix[col]
				distances[row] += np.full((N,), distances[row, col]) * adj_matrix[col]	# BROADCAST ERROR!

	return distances

# EVA Program that computes distances between each elements.
def computeDistances(inputs, n, vec_size, config, random_index):
	computeDistancesProgram = EvaProgram("Compute distances", vec_size=vec_size)
	with computeDistancesProgram:
		input = {'zero': Input('zero', is_encrypted=True)}
		for i in range(n):
			for j in range(n):
				key = f"n_{i}_{j}"
				input[key] = Input(key)
		
		output_matrix = computeDistancesAnalytics(input, n)
		N = n * n
		for i in range(N):
			for j in range(N):
				key = f"n_{i}_{j}"
				Output(key, output_matrix[i,j])
	inputs['zero'] = [0] * vec_size	# https://github.com/microsoft/SEAL/issues/200
	
	prog = computeDistancesProgram
	prog.set_output_ranges(60)
	prog.set_input_scales(60)

	# Compilation
	start = timeit.default_timer()
	compiler = CKKSCompiler(config=config)
	compiled_multfunc, params, signature = compiler.compile(prog)
	compileTime = (timeit.default_timer() - start) * 1000.0 #ms

	# Key Generation
	start = timeit.default_timer()
	public_ctx, secret_ctx = generate_keys(params)
	keyGenerationTime = (timeit.default_timer() - start) * 1000.0 #ms
	
	# Encryption
	start = timeit.default_timer()
	encInputs = public_ctx.encrypt(inputs, signature)
	encryptionTime = (timeit.default_timer() - start) * 1000.0 #ms

	# Execution
	start = timeit.default_timer()
	encOutputs = public_ctx.execute(compiled_multfunc, encInputs)
	executionTime = (timeit.default_timer() - start) * 1000.0 #ms

	# Decryption
	start = timeit.default_timer()
	outputs = secret_ctx.decrypt(encOutputs, signature)
	decryptionTime = (timeit.default_timer() - start) * 1000.0 #ms

	# Evaluation
	start = timeit.default_timer()
	reference = evaluate(compiled_multfunc, inputs)
	referenceExecutionTime = (timeit.default_timer() - start) * 1000.0 #ms

	# Approximation
	mse = valuation_mse(outputs, reference) # since CKKS does approximate computations, this is an important measure that depicts the amount of error

	timings = {'compile': compileTime,
			   'keyGeneration': keyGenerationTime,
			   'encryption': encryptionTime,
			   'execution': executionTime,
			   'decryption': decryptionTime,
			   'referenceExecution': referenceExecutionTime,
			   'mse': mse
			   }
	printComputeDistances(inputs, outputs, reference, n, random_index, timings, verbose=VERBOSE)
	return outputs, timings

# Count the islands
def countIslands(inputs, distances, n, vec_size, random_index, verbose=False):
	N = n * n
	counts = []
	for vec in range(vec_size):
		input_matrix = np.zeros(shape=(n,n))
		distance_matrix = np.zeros(shape=(N,N))

		for i in range(N):
			for j in range(N):
				key = f"n_{i}_{j}"
				if i < n and j < n:
					input_matrix[i,j] = inputs[key][vec]
				distance_matrix[i,j] = distances[key][vec]
		
		# round the values to closest integers
		input_matrix = np.rint(input_matrix).astype(int)
		distance_matrix = np.rint(distance_matrix).astype(int)

		indices = {}
		islands = set()
		for d_row in range(N):	# row of distance matrix
			i_row = d_row // n	# row of input matrix
			i_col = d_row % n	# column of input matrix

			if input_matrix[i_row,i_col] == 1:
				# all reachable (i,j) elements from (i_row,i_col)
				reachable_elements = [(x//n, x%n) for x in np.argwhere(distance_matrix[d_row] != 0).ravel() if (x != d_row) and (input_matrix[x//n,x%n] == 1)]
				indices[(i_row,i_col)] = reachable_elements

				if not (set(indices[(i_row,i_col)]) & islands):
					islands.add((i_row,i_col))
					counts.append(len(islands))

		if vec == random_index and verbose:
			print("\nCOUNT THE ISLANDS:")
			print("Input Matrix:")
			print(input_matrix)
			print(f"Islands: {sorted(islands)} and Island Count: {len(islands)}")
	
	return counts

# Repeat the experiments and show averages with confidence intervals
def simulate(n, vec_size):
	config = {}
	config['balance_reductions'] = "true"
	config['rescaler'] = "always"
	config['lazy_relinearize'] = "true"
	config['security_level'] = "128"
	config['warn_vec_size'] = "false"

	# Prepare input matrices similar to EVA Input
	inputs = prepareInputs(n, vec_size)

	# select random index from vector size, used only in print functions
	random_index = randint(0, vec_size-1)

	# Reduce the number of ones in the input matrix with EVA Program.
	reducedInputs, timings_reduced = reduceOnes(inputs, n, vec_size, config, random_index)

	# Get distances between each elements with EVA Program.
	distances, timings_distances = computeDistances(inputs, n, vec_size, config, random_index)

	countIslands(reducedInputs, distances, n, vec_size, random_index, True)

	return timings_reduced, timings_distances

if __name__ == "__main__":
	simcnt = 100	# The number of simulation runs for each n and vec_size
	
	os.makedirs("results", exist_ok=True)	# create results directory if not exists
	results_reduce_file = "results/parallel_reduce.csv"
	results_distances_file = "results/parallel_distances.csv"

	with open(results_reduce_file, 'w') as  f:
		f.write("n,VecSize,sim,CompileTime,KeyGenerationTime,EncryptionTime,ExecutionTime,DecryptionTime,ReferenceExecutionTime,Mse\n")
		f.close()
	with open(results_distances_file, 'w') as  f:
		f.write("n,VecSize,sim,CompileTime,KeyGenerationTime,EncryptionTime,ExecutionTime,DecryptionTime,ReferenceExecutionTime,Mse\n")
		f.close()
	
	print("Simulation campaing started:")
	N_LIST = [1,2]
	VEC_POWERS = [10,11,12,13,14]
	
	for n in N_LIST:
		for vec_pow in VEC_POWERS:
			vec_size = 2**vec_pow
			for sim in range(simcnt):
				print(f"[{sim+1}/{simcnt}] - n:{n}, vec_size:{vec_size}")
				# Call the simulator
				timings_reduce, timings_distances = simulate(n, vec_size)
				
				with open(results_reduce_file, "a") as f:
					result_str = f"{n},{vec_size},{sim},{timings_reduce['compile']},{timings_reduce['keyGeneration']},{timings_reduce['encryption']},{timings_reduce['execution']},{timings_reduce['decryption']},{timings_reduce['referenceExecution']},{timings_reduce['mse']}"
					if VERBOSE: print(result_str)
					f.write(result_str + "\n")
				
				with open(results_distances_file, "a") as f:
					result_str = f"{n},{vec_size},{sim},{timings_distances['compile']},{timings_distances['keyGeneration']},{timings_distances['encryption']},{timings_distances['execution']},{timings_distances['decryption']},{timings_distances['referenceExecution']},{timings_distances['mse']}"
					if VERBOSE: print(result_str)
					f.write(result_str + "\n")