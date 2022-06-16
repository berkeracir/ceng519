from math import ceil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

labels = ['Vectorized', 'Parallelized', 'P. w/o. Loops']
reduce_files = ['vectorized_reduce.csv', 'parallel_reduce.csv', 'parallel_loop_reduce.csv']
distances_files = ['vectorized_distances.csv', 'parallel_distances.csv', 'parallel_loop_distances.csv']

os.makedirs('plot_by_totalruntime', exist_ok=True)	# create directory if not exists
dir_name = 'plot_by_totalruntime/'

# TOTAL TIMINGS by ONE INPUT
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,4), constrained_layout=True)
fig.suptitle('Total Running Times of EVA Programs for Only One Input')

# REDUCE FILES
for i in range(len(reduce_files)):
	reduce_file = reduce_files[i]

	# Header: n VecSize sim CompileTime KeyGenerationTime EncryptionTime ExecutionTime DecryptionTime ReferenceExecutionTime MSE
	df = pd.read_csv(reduce_file).drop(columns=['sim'])
	if i != 0:
		df = df[df['VecSize'] == 1]

	df['n'] = df['n'] * df['n']
	df['total'] = df['CompileTime'] + df['KeyGenerationTime'] + df['EncryptionTime'] + df['ExecutionTime'] + df['DecryptionTime'] + df['ReferenceExecutionTime']
	gb = df.groupby(['n'])
	gb_values = list(gb.groups)
	mean = gb.mean()

	axs[0].plot(gb_values, mean['total'].to_numpy(), label=labels[i])

# DISTANCES FILES
for i in range(len(distances_files)):
	distances_file = distances_files[i]

	# Header: n VecSize sim CompileTime KeyGenerationTime EncryptionTime ExecutionTime DecryptionTime ReferenceExecutionTime MSE
	df = pd.read_csv(distances_file).drop(columns=['sim'])
	if i != 0:
		df = df[df['VecSize'] == 1]

	df['n'] = df['n'] * df['n']
	df['total'] = (df['CompileTime'] + df['KeyGenerationTime'] + df['EncryptionTime'] + df['ExecutionTime'] + df['DecryptionTime'] + df['ReferenceExecutionTime'])/(60*1000) #minute
	if i != 1:
		df['total'] = df['total'] * (df['n']**2/2).apply(np.ceil) * df['n']**2
	gb = df.groupby(['n'])
	gb_values = list(gb.groups)
	mean = gb.mean()

	axs[1].plot(gb_values, mean['total'].to_numpy(), label=labels[i])

axs[0].set_title('Reduce Ones')
axs[1].set_title('Compute Distances')

for i in range(2):
	axs[i].legend(loc='best')
axs[0].set_ylabel('Total Time (ms)')	
axs[1].set_ylabel('Total Time (minute)')	
plt.setp(axs, xlabel='Input Matrix Size', xticks=np.arange(1,6)**2)
plt.savefig(dir_name + 'total_runtimes.png')
plt.clf()	# clear the saved figure


# AMORTIZED TOTAL TIMINGS
fig_amortized, axs_amortized = plt.subplots(nrows=1, ncols=2, figsize=(8,4), constrained_layout=True)
fig_amortized.suptitle('Amortized Total Running Times of EVA Programs')

# REDUCE FILES
for i in range(len(reduce_files)):
	reduce_file = reduce_files[i]

	# Header: n VecSize sim CompileTime KeyGenerationTime EncryptionTime ExecutionTime DecryptionTime ReferenceExecutionTime MSE
	df = pd.read_csv(reduce_file).drop(columns=['sim'])

	df['n'] = df['n'] * df['n']
	df['total'] = df['CompileTime'] + df['KeyGenerationTime'] + df['EncryptionTime'] + df['ExecutionTime'] + df['DecryptionTime'] + df['ReferenceExecutionTime']
	if i != 0:
		df['total'] = df['total']/df['VecSize']
	gb = df.groupby(['n'])
	gb_values = list(gb.groups)
	mean = gb.mean()

	axs_amortized[0].plot(gb_values, mean['total'].to_numpy(), label=labels[i])

# DISTANCES FILES
for i in range(len(distances_files)):
	distances_file = distances_files[i]

	# Header: n VecSize sim CompileTime KeyGenerationTime EncryptionTime ExecutionTime DecryptionTime ReferenceExecutionTime MSE
	df = pd.read_csv(distances_file).drop(columns=['sim'])

	df['n'] = df['n'] * df['n']
	df['total'] = (df['CompileTime'] + df['KeyGenerationTime'] + df['EncryptionTime'] + df['ExecutionTime'] + df['DecryptionTime'] + df['ReferenceExecutionTime'])/(60*1000) #minute
	if i != 1:
		df['total'] = df['total'] * (df['n']**2/2).apply(np.ceil) * df['n']**2
	if i != 0:
		df['total'] = df['total']/df['VecSize']
	gb = df.groupby(['n'])
	gb_values = list(gb.groups)
	mean = gb.mean()

	axs_amortized[1].plot(gb_values, mean['total'].to_numpy(), label=labels[i])

axs_amortized[0].set_title('Reduce Ones')
axs_amortized[1].set_title('Compute Distances')

for i in range(2):
	axs_amortized[i].legend(loc='best')
axs_amortized[0].set_ylabel('Total Time (ms)')	
axs_amortized[1].set_ylabel('Total Time (minute)')	
plt.setp(axs_amortized, xlabel='Input Matrix Size', xticks=np.arange(1,6)**2)
plt.savefig(dir_name + 'amortized_total_runtimes.png')