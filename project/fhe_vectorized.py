import timeit
from math import ceil, pow, log2
import numpy as np
import os

from eva import EvaProgram, Input, Output, evaluate
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
def adjacencyMatrix(arr, n):
	N = n * n	# matrix size is squared
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
						result[row, col] = arr[i, j]

	return result

def printReduceOnes(inputs, outputs, reference, n, timings, verbose=False):
	if verbose:
		print(f"\nREDUCE ONES:\n" +
				f"-Compile Time: {timings['compile']}\n" +
				f"-Key Generation Time: {timings['keyGeneration']}\n" +
				f"-Encryption Time: {timings['encryption']}\n" + 
				f"-Execution Time: {timings['execution']}\n" +
				f"-Decryption Time: {timings['decryption']}\n" +
				f"-Reference Execution Time: {timings['referenceExecution']}\n" +
				f"-MSE: {timings['mse']}")

		input_matrix = np.array(inputs['input'])[:(n+2)**2].reshape((n+2,n+2))[1:n+1,1:n+1]
		reduced_input_matrix = np.array(outputs['reduced'])[:(n+2)**2].reshape((n+2,n+2))[1:n+1,1:n+1]
		reference_matrix = np.array(reference['reduced'])[:(n+2)**2].reshape((n+2,n+2))[1:n+1,1:n+1]
	
		input_matrix = np.rint(input_matrix).astype(int)
		reduced_input_matrix = np.rint(reduced_input_matrix).astype(int)
		reference_matrix = np.rint(reference_matrix).astype(int)
		print("Input Matrix:")
		print(input_matrix)
		print("Reduced Matrix:")
		print(reduced_input_matrix)
		print(f"Reduced Matrix {'==' if (np.array_equal(reduced_input_matrix, reference_matrix)) else '!='} Reference")

def printComputeDistances(inputs, outputs, reference, n, verbose=False):
	if verbose:		
		adjaceceny_matrix = np.array(inputs['adjacency'])[:n**4].reshape((n**2,n**2))
		distance_matrix = np.array(outputs['distance'])[:n**4].reshape((n**2,n**2))
		extended_matrix = np.array(inputs['extended'])[:n**4].reshape((n**2,n**2))
		reference_matrix = np.array(reference['distance'])[:n**4].reshape((n**2,n**2))

		adjaceceny_matrix = np.rint(adjaceceny_matrix).astype(int)
		distance_matrix = np.rint(distance_matrix).astype(int)
		extended_matrix = np.rint(extended_matrix).astype(int)
		reference_matrix = np.rint(reference_matrix).astype(int)
		print("Input Adjacency Matrix:")
		print(adjaceceny_matrix)
		print("Distance Matrix:")
		print(distance_matrix)
		print("Extended Matrix:")
		print(extended_matrix)
		print(f"Distance Matrix {'==' if (np.array_equal(distance_matrix, reference_matrix)) else '!='} Reference")

def printComputeDistancesTimings(timings, verbose=False):
	if verbose:
		call_count = len(timings['compile'])
		print(f"\nCOMPUTE DISTANCES (summed):\n" +
				f"-Total Call Count: {call_count}\n" +
				f"-Compile Times: {sum(timings['compile'])}\n" +
				f"-Key Generation Times: {sum(timings['keyGeneration'])}\n" +
				f"-Encryption Times: {sum(timings['encryption'])}\n" + 
				f"-Execution Times: {sum(timings['execution'])}\n" +
				f"-Decryption Times: {sum(timings['decryption'])}\n" +
				f"-Reference Execution Times: {sum(timings['referenceExecution'])}\n" +
				f"-MSE (avg.): {sum(timings['mse'])/call_count}")

# Prepare input matrix of size (vec_size) where there are (n*n) random 0s and 1s for representing padded input matrix. 
def prepareInputs(n, vec_size):
	inputs = {}

	# Create a matrix of size (vec_size) with (n*n) random 0s and 1s.
	input_matrix = np.zeros(shape=(vec_size))
	padded_input = np.zeros(shape=(n+2,n+2))
	padded_input[1:n+1,1:n+1] = np.random.randint(2, size=(n,n))
	input_matrix[:(n+2)*(n+2)] = padded_input.ravel()
	inputs['input'] = list(input_matrix)

	return inputs

# Prepare adjacency inputs
def prepareAdjacencyInputs(inputs, adjacency_matrix, distance_input, shift, n, vec_size):
	adjacencyInputs = {}
	N = n * n

	# If adjacency matrix is not given, convert adjacency matrix of the inputs
	if adjacency_matrix is None:
		input_matrix = np.array(inputs['input'])[:(n+2)**2].reshape((n+2,n+2))[1:n+1,1:n+1]
		input_matrix = np.rint(input_matrix).astype(int)

		# Get the adjacency matrix from the input matrix
		adjacency_matrix = adjacencyMatrix(input_matrix, n)
		# print("Adjacency Matrix")
		# print(adjacency_matrix)
	
	adjacency_input = np.zeros(shape=(vec_size))
	adjacency_input[:N**2] = np.tile(adjacency_matrix.ravel()[N*shift:N*(shift+1)], N)
	adjacencyInputs['adjacency'] = list(adjacency_input)

	# If distance input is not given, initialized it to the adjacency input.
	if distance_input is None:
		distance_input = np.zeros(shape=(vec_size))
		distance_input[:N**2] = adjacency_matrix.ravel()
	else:
		# If there are already distance input, then round its' values to nearest integers.
		# Round the distance values in order to reset the approximation to some extend. 
		distance_input = np.rint(np.array(distance_input))
		distance_input[distance_input != 0] = 1	# In order to prevent overflowing issues.
	adjacencyInputs['distance'] = list(distance_input)
	
	extended_input = np.zeros(shape=(vec_size))
	for i in range(N):
		extended_input[N*i:N*(i+1)] = np.tile(distance_input[N*i+shift], N)
	adjacencyInputs['extended'] = list(extended_input)

	return adjacencyInputs, adjacency_matrix

# Analytic service that reduces the number of ones.
def reduceOnesAnalytics(input, n):
	element = input['input']
	l_element = input['input'] >> 1			# left element of the element
	ul_element = input['input'] >> (n+2) + 1	# upper left element of the element
	u_element = input['input'] >> (n+2)		# upper element of the element
	ur_element = input['input'] >> (n+2) - 1	# upper right element of the element

	retval = element * (element + l_element + ul_element + u_element + ur_element)
	return retval

# EVA Program that reduces the number of ones.
def reduceOnes(inputs, n, vec_size, config):
	reduceOnesProgram = EvaProgram("Reduce ones", vec_size=vec_size)
	with reduceOnesProgram:
		input = {'input': Input('input')}
		output = reduceOnesAnalytics(input, n)
		Output('reduced', output)
	
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
	printReduceOnes(inputs, outputs, reference, n, timings, verbose=VERBOSE)
	return outputs, timings

# Analytic service that computes distances between each elements.
def computeDistancesAnalytics(input):
	adjacency = input['adjacency']
	distance = input['distance']
	extended = input['extended']

	distance += extended * adjacency

	return distance

# EVA Program that computes distances between each elements.
def computeDistances(inputs, n, shift, vec_size, config, verbose=False):
	computeDistancesProgram = EvaProgram("Compute distances", vec_size=vec_size)
	with computeDistancesProgram:
		input = {'adjacency': Input('adjacency'),
				 'distance': Input('distance'),
				 'extended': Input('extended')}
		distance = computeDistancesAnalytics(input)
		Output('distance', distance)
	
	prog = computeDistancesProgram
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
	printComputeDistances(inputs, outputs, reference, n, verbose=verbose)
	return outputs, timings

# Count the islands
def countIslands(inputs, outputs, n, verbose=False):
	N = n * n

	input_matrix = np.array(inputs['reduced'])[:(n+2)**2].reshape((n+2,n+2))[1:n+1,1:n+1]
	input_matrix = np.rint(input_matrix).astype(int)
	distance_matrix = np.array(outputs['distance'])[:(n**4)].reshape((n**2,n**2))
	distance_matrix = np.rint(distance_matrix)

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

	if verbose:
		print("\nCOUNT THE ISLANDS:")
		print("Input Matrix:")
		print(input_matrix)
		print(f"Islands: {sorted(islands)} and Island Count: {len(islands)}")
	
	return len(islands)

# Repeat the experiments and show averages with confidence intervals
def simulate(n, vec_sizes):
	N = n * n
	config = {}
	config['balance_reductions'] = "true"
	config['rescaler'] = "always"
	config['lazy_relinearize'] = "true"
	config['security_level'] = "128"
	config['warn_vec_size'] = "false"

	# Prepare input matrix similar to EVA Input
	inputs = prepareInputs(n, vec_sizes[0])

	# Reduce the number of ones in the input matrix with EVA Program.
	reduced_inputs, timings_reduced = reduceOnes(inputs, n, vec_sizes[0], config)

	# Get distances between each elements with EVA Program.
	timings_distances = {}
	adjacency_matrix, distance_input = None, None
	for repeat in range(ceil(N/2)):
		for shift in range(N):
			verbose = VERBOSE if (repeat == ceil(N/2)-1 and shift == N-1) else False

			# Prepare input adjacency matrix similar to EVA Input
			adjacencyInputs, adjacency_matrix = prepareAdjacencyInputs(inputs, adjacency_matrix, distance_input, shift, n, vec_sizes[1])

			outputs, timings = computeDistances(adjacencyInputs, n, shift, vec_sizes[1], config, verbose)
			distance_input = outputs['distance']

			for key in timings.keys():
				if key in timings_distances.keys():
					timings_distances[key].append(timings[key])
				else:
					timings_distances[key] = [timings[key]]
	
	printComputeDistancesTimings(timings_distances, VERBOSE)

	countIslands(reduced_inputs, outputs, n, True)

	return timings_reduced, timings_distances

if __name__ == "__main__":
	simcnt = 100	# The number of simulation runs for each n
	
	os.makedirs("results", exist_ok=True)	# create results directory if not exists
	results_reduce_file = "results/vectorized_reduce.csv"
	results_distances_file = "results/vectorized_distances.csv"

	with open(results_reduce_file, 'w') as  f:
		f.write("n,VecSize,sim,CompileTime,KeyGenerationTime,EncryptionTime,ExecutionTime,DecryptionTime,ReferenceExecutionTime,MSE\n")
		f.close()
	with open(results_distances_file, 'w') as  f:
		f.write("n,VecSize,sim,CompileTime,KeyGenerationTime,EncryptionTime,ExecutionTime,DecryptionTime,ReferenceExecutionTime,MSE\n")
		f.close()
	
	print("Simulation campaing started:")
	N_LIST = [1,2,3,4,5]
	
	for n in N_LIST:
		n_reduce = pow(n+2,2)	# +2 comes from paddings
		vec_size_reduce = int(pow(2,ceil(log2(n_reduce))))
		n_distances = pow(n,4)
		vec_size_distances = int(pow(2,ceil(log2(n_distances))))
		vec_sizes = [vec_size_reduce, vec_size_distances]

		for sim in range(simcnt):
			print(f"[{sim+1}/{simcnt}] - n:{n}, vec_sizes:{vec_size_reduce}/{vec_size_distances}")
			# Call the simulator
			timings_reduce, timings_distances = simulate(n, vec_sizes)
			
			with open(results_reduce_file, "a") as f:
				result_str = f"{n},{vec_size_reduce},{sim},{timings_reduce['compile']},{timings_reduce['keyGeneration']},{timings_reduce['encryption']},{timings_reduce['execution']},{timings_reduce['decryption']},{timings_reduce['referenceExecution']},{timings_reduce['mse']}"
				if VERBOSE: print(result_str)
				f.write(result_str + "\n")
			
			with open(results_distances_file, "a") as f:
				for i in range(len(timings_distances['compile'])):
					result_str = f"{n},{vec_size_distances},{sim},{timings_distances['compile'][i]},{timings_distances['keyGeneration'][i]},{timings_distances['encryption'][i]},{timings_distances['execution'][i]},{timings_distances['decryption'][i]},{timings_distances['referenceExecution'][i]},{timings_distances['mse'][i]}"
					if VERBOSE: print(result_str)
					f.write(result_str + "\n")