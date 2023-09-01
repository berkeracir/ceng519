from math import ceil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.interpolate import make_interp_spline

plt.rcParams["figure.figsize"] = [4, 4]
plt.rcParams["figure.autolayout"] = True
plt.rcParams['figure.dpi'] = 250

labels = ['Vectorized'] # ['Vectorized', 'Parallelized', 'P. w/o. Loops']
reduce_files = [] # ['vectorized_reduce.csv', 'parallel_reduce.csv', 'parallel_loop_reduce.csv']
distances_files = ['vectorized_distances.csv'] # ['vectorized_distances.csv', 'parallel_distances.csv', 'parallel_loop_distances.csv']

os.makedirs('plot_new_new', exist_ok=True)	# create directory if not exists
dir_name = 'plot_new_new/'

# TOTAL TIMINGS by ONE INPUT
# fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,4), constrained_layout=True)
# fig.suptitle('Total Running Times of EVA Programs for Only One Input')

# # REDUCE FILES
# for i in range(len(reduce_files)):
# 	reduce_file = reduce_files[i]

# 	# Header: n VecSize sim CompileTime KeyGenerationTime EncryptionTime ExecutionTime DecryptionTime ReferenceExecutionTime MSE
# 	df = pd.read_csv(reduce_file).drop(columns=['sim'])
# 	if i != 0:
# 		df = df[df['VecSize'] == 1]

# 	df['n'] = df['n'] * df['n']
# 	df['total'] = df['CompileTime'] + df['KeyGenerationTime'] + df['EncryptionTime'] + df['ExecutionTime'] + df['DecryptionTime'] + df['ReferenceExecutionTime']
# 	gb = df.groupby(['n'])
# 	gb_values = list(gb.groups)
# 	mean = gb.mean()

# 	axs[0].plot(gb_values, mean['total'].to_numpy(), label=labels[i])

# DISTANCES FILES
for i in range(len(distances_files)):
	distances_file = distances_files[i]

	# Header: n VecSize sim CompileTime KeyGenerationTime EncryptionTime ExecutionTime DecryptionTime ReferenceExecutionTime MSE
	df = pd.read_csv(distances_file).drop(columns=['sim'])
	if i != 0:
		df = df[df['VecSize'] == 1]

	df['n'] = df['n'] * df['n']
	df['total'] = (df['CompileTime'] + df['KeyGenerationTime'] + df['EncryptionTime'] + df['ExecutionTime'] + df['DecryptionTime'] + df['ReferenceExecutionTime'])/(1000) # second
	if i != 1:
		df['total'] = df['total'] * (df['n']/2).apply(np.ceil) * df['n']
	gb = df.groupby(['n'])
	gb_values = list(gb.groups)
	mean = gb.mean()

	x = gb_values
	y = mean['total'].to_numpy()
	X_Y_Spline = make_interp_spline(x, y)
	X_ = np.linspace(1, 25, 500)
	Y_ = X_Y_Spline(X_)

	plt.plot(X_, Y_, '-', label=labels[i], color='indigo')
	plt.plot(gb_values, mean['total'].to_numpy(), 'o', label=labels[i], color='indigo')

# axs[0].set_title('Reduce Ones')
# axs[1].set_title('Compute Distances')

# for i in range(2):
# 	axs[i].legend(loc='best')
# axs[0].set_ylabel('Total Time (ms)')	
# axs[1].set_ylabel('Total Time (minute)')	
# plt.setp(axs, xlabel='Input Matrix Size', xticks=np.arange(1,6)**2)
# plt.savefig(dir_name + 'total_runtimes.png')
# plt.clf()	# clear the saved figure


plt.xlabel('Girdi Matrisi Boyutu')
plt.ylabel('Zaman (sn)')
plt.xticks(np.arange(1,6)**2, [f"{int(N**0.5)}x{int(N**0.5)}" for N in np.arange(1,6)**2])
location = (0.01,0.55)
# plt.legend(bars, bar_names)#, loc='upper left')
plt.xlim(0, 26)
# if i == 0:
# 	plt.ylim(0,100)
# else:
# 	plt.ylim(bottom=0)
plt.savefig(dir_name + 'total_runtimes.png')
plt.clf()	# clear the saved figure