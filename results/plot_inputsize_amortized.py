import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

labels = ['Vectorized', 'Parallelized', 'P. w/o. Loops']
reduce_files = ['vectorized_reduce.csv', 'parallel_reduce.csv', 'parallel_loop_reduce.csv']
distances_files = ['vectorized_distances.csv', 'parallel_distances.csv', 'parallel_loop_distances.csv']

os.makedirs('plot_by_inputsize', exist_ok=True)	# create directory if not exists
dir_name = 'plot_by_inputsize/'

width = 0.3
# REDUCE TIMINGS
fig_reduce, axs_reduce = plt.subplots(nrows=2, ncols=3, figsize=(12,6), constrained_layout=True)
fig_reduce.suptitle('Amortized Times for Reduce Ones EVA Program')

# REDUCE FILES
for i in range(len(reduce_files)):
	distances_file = reduce_files[i]

	# Header: n VecSize sim CompileTime KeyGenerationTime EncryptionTime ExecutionTime DecryptionTime ReferenceExecutionTime MSE
	df = pd.read_csv(distances_file).drop(columns=['sim'])
	df = df[((df['n'] == 1) | (df['n'] == 2) | (df['n'] == 3))]
	if i != 0:
		df = df[df['VecSize'] == 1024]
		df['CompileTime'] = df['CompileTime'] / df['VecSize']
		df['KeyGenerationTime'] = df['KeyGenerationTime'] / df['VecSize']
		df['EncryptionTime'] = df['EncryptionTime'] / df['VecSize']
		df['ExecutionTime'] = df['ExecutionTime'] / df['VecSize']
		df['DecryptionTime'] = df['DecryptionTime'] / df['VecSize']
		df['ReferenceExecutionTime'] = df['ReferenceExecutionTime'] / df['VecSize']

	df['n'] = df['n'] * df['n']
	gb = df.groupby(['n'])
	gb_values = list(gb.groups)
	mean = gb.mean()
	std = gb.std()
	x_pos = np.arange(len(gb_values))

	axs_reduce[0,0].bar(x_pos + width*(i-1), mean['CompileTime'], width, yerr=std['CompileTime'], align='center', capsize=3, label=labels[i])
	axs_reduce[0,1].bar(x_pos + width*(i-1), mean['KeyGenerationTime'], width, yerr=std['KeyGenerationTime'], align='center', capsize=3, label=labels[i])
	axs_reduce[0,2].bar(x_pos + width*(i-1), mean['EncryptionTime'], width, yerr=std['EncryptionTime'], align='center', capsize=3, label=labels[i])
	axs_reduce[1,0].bar(x_pos + width*(i-1), mean['ExecutionTime'], width, yerr=std['ExecutionTime'], align='center', capsize=3, label=labels[i])
	axs_reduce[1,1].bar(x_pos + width*(i-1), mean['DecryptionTime'], width, yerr=std['DecryptionTime'], align='center', capsize=3, label=labels[i])
	axs_reduce[1,2].bar(x_pos + width*(i-1), mean['ReferenceExecutionTime'], width, yerr=std['ReferenceExecutionTime'], align='center', capsize=3, label=labels[i])

axs_reduce[0, 0].set_title('Compile Times')
axs_reduce[0, 1].set_title('Key Generation Times')
axs_reduce[0, 2].set_title('Encryption Times')
axs_reduce[1, 0].set_title('Execution Times')
axs_reduce[1, 1].set_title('Decryption Times')
axs_reduce[1, 2].set_title('Reference Execution Times')

for i in range(2):
	for j in range(3):
		axs_reduce[i,j].legend(loc='best')
		axs_reduce[i,j].set_ylim(bottom=0)
plt.setp(axs_reduce, xlabel='Input Matrix Size', ylabel='Time (ms)', xticks=x_pos, xticklabels=gb_values, xlim=(x_pos[0]-1.5*width, x_pos[-1]+1.5*width))
# plt.show()
plt.savefig(dir_name + 'amortized_reduce_times.png')
plt.clf()	# clear the saved figure

# DISTANCES TIMINGS
fig_distances, axs_distances = plt.subplots(nrows=2, ncols=3, figsize=(12,6), constrained_layout=True)
fig_distances.suptitle('Amortized Times for Compute Distances EVA Program')

# DISTANCES FILES
for i in range(len(distances_files)):
	distances_file = distances_files[i]

	# Header: n VecSize sim CompileTime KeyGenerationTime EncryptionTime ExecutionTime DecryptionTime ReferenceExecutionTime MSE
	df = pd.read_csv(distances_file).drop(columns=['sim'])
	df = df[((df['n'] == 1) | (df['n'] == 2) | (df['n'] == 3))]
	if i != 0:
		df = df[df['VecSize'] == 1024]
		df['CompileTime'] = df['CompileTime'] / df['VecSize']
		df['KeyGenerationTime'] = df['KeyGenerationTime'] / df['VecSize']
		df['EncryptionTime'] = df['EncryptionTime'] / df['VecSize']
		df['ExecutionTime'] = df['ExecutionTime'] / df['VecSize']
		df['DecryptionTime'] = df['DecryptionTime'] / df['VecSize']
		df['ReferenceExecutionTime'] = df['ReferenceExecutionTime'] / df['VecSize']

	df['n'] = df['n'] * df['n']
	gb = df.groupby(['n'])
	gb_values = list(gb.groups)
	mean = gb.mean()
	std = gb.std()
	x_pos = np.arange(len(gb_values))

	axs_distances[0,0].bar(x_pos + width*(i-1), mean['CompileTime'], width, yerr=std['CompileTime'], align='center', capsize=3, label=labels[i])
	axs_distances[0,1].bar(x_pos + width*(i-1), mean['KeyGenerationTime'], width, yerr=std['KeyGenerationTime'], align='center', capsize=3, label=labels[i])
	axs_distances[0,2].bar(x_pos + width*(i-1), mean['EncryptionTime'], width, yerr=std['EncryptionTime'], align='center', capsize=3, label=labels[i])
	axs_distances[1,0].bar(x_pos + width*(i-1), mean['ExecutionTime'], width, yerr=std['ExecutionTime'], align='center', capsize=3, label=labels[i])
	axs_distances[1,1].bar(x_pos + width*(i-1), mean['DecryptionTime'], width, yerr=std['DecryptionTime'], align='center', capsize=3, label=labels[i])
	axs_distances[1,2].bar(x_pos + width*(i-1), mean['ReferenceExecutionTime'], width, yerr=std['ReferenceExecutionTime'], align='center', capsize=3, label=labels[i])

axs_distances[0, 0].set_title('Compile Times')
axs_distances[0, 1].set_title('Key Generation Times')
axs_distances[0, 2].set_title('Encryption Times')
axs_distances[1, 0].set_title('Execution Times')
axs_distances[1, 1].set_title('Decryption Times')
axs_distances[1, 2].set_title('Reference Execution Times')

for i in range(2):
	for j in range(3):
		axs_distances[i,j].legend(loc='best')
		axs_distances[i,j].set_ylim(bottom=0)
plt.setp(axs_distances, xlabel='Input Matrix Size', ylabel='Time (ms)', xticks=x_pos, xticklabels=gb_values, xlim=(x_pos[0]-1.5*width, x_pos[-1]+1.5*width))
plt.savefig(dir_name + 'amortized_distances_times.png')