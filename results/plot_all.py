import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams["figure.figsize"] = [8, 4]
plt.rcParams["figure.autolayout"] = True
plt.rcParams['figure.dpi'] = 250

titles = ['Vectorized Implementation'] # ['Vectorized Implementation', 'Parallelized Implementation', 'Parallelized Implementation without Loops']
reduce_files = [] # ['vectorized_reduce.csv', 'parallel_reduce.csv', 'parallel_loop_reduce.csv']
distances_files = ['vectorized_distances.csv'] # ['vectorized_distances.csv', 'parallel_distances.csv', 'parallel_loop_distances.csv']

os.makedirs('plot_new', exist_ok=True)	# create directory if not exists
dir_name = 'plot_new/'

# REDUCE FILES
for i in range(len(reduce_files)):
	reduce_file = reduce_files[i]
	title = titles[i]

	# Header: n VecSize sim CompileTime KeyGenerationTime EncryptionTime ExecutionTime DecryptionTime ReferenceExecutionTime MSE
	df = pd.read_csv(reduce_file).drop(columns=['sim'])
	df['n'] = df['n'] * df['n']
	gb = df.groupby(['n','VecSize'])
	gb_values = list(gb.groups)
	mean = gb.mean()
	std = gb.std()
	x_pos = np.arange(len(gb_values))

	plt.figure(figsize=(1 * len(gb_values), 4))

	width = 0.15
	bar1 = plt.bar(x_pos - width*2.5, mean['CompileTime'], width, yerr=std['CompileTime'], align='center', capsize=3)
	bar2 = plt.bar(x_pos - width*1.5, mean['KeyGenerationTime'], width, yerr=std['KeyGenerationTime'], align='center', capsize=3)
	bar3 = plt.bar(x_pos - width*0.5, mean['EncryptionTime'], width, yerr=std['EncryptionTime'], align='center', capsize=3)
	bar4 = plt.bar(x_pos + width*0.5, mean['ExecutionTime'], width, yerr=std['ExecutionTime'], align='center', capsize=3)
	bar5 = plt.bar(x_pos + width*1.5, mean['DecryptionTime'], width, yerr=std['DecryptionTime'], align='center', capsize=3)
	bar6 = plt.bar(x_pos + width*2.5, mean['ReferenceExecutionTime'], width, yerr=std['ReferenceExecutionTime'], align='center', capsize=3)
	bars = [bar1, bar2, bar3, bar4, bar5, bar6]
	bar_names = ['Compile', 'Key Generation', 'Encryption', 'Execution', 'Decryption', 'Reference Execution']

	plt.xlabel('Input Matrix Size, Vector Size')
	plt.ylabel('Time (ms)')
	plt.title(f'Times for Reduced Ones ({title})')
	plt.xticks(x_pos, gb_values, rotation = 0)
	plt.legend(bars, bar_names, loc='upper left')
	plt.xlim(x_pos[0] - width*3, x_pos[-1] + width*3)
	plt.ylim(0,100)
	plt.savefig(dir_name + reduce_file.replace('.csv', '_times.png'))
	plt.clf()	# clear the saved figure

	if i != 0:
		plt.figure(figsize=(0.7 * len(gb_values), 4))

	width = 0.8
	bar_mse = plt.bar(x_pos, mean['MSE'], width, yerr=std['MSE'], align='center', capsize=3, color='indianred')

	plt.xlabel('Input Matrix Size, Vector Size')
	plt.ylabel('MSE Score')
	plt.title(f'MSE Scores for Reduce Ones ({title})')
	plt.xticks(x_pos, gb_values, rotation = 0)
	plt.xlim(x_pos[0] - width/2, x_pos[-1] + width/2)
	plt.ylim(bottom=0)
	plt.savefig(dir_name + reduce_file.replace('.csv', '_mse.png'))

# DISTANCES FILES
for i in range(len(distances_files)):
	distances_file = distances_files[i]
	title = titles[i]

	# Header: n VecSize sim CompileTime KeyGenerationTime EncryptionTime ExecutionTime DecryptionTime ReferenceExecutionTime MSE
	df = pd.read_csv(distances_file).drop(columns=['sim'])
	df['n'] = df['n'] * df['n']
	gb = df.groupby(['n','VecSize'])
	gb_values = list(gb.groups)
	mean = gb.mean()
	std = gb.std()
	x_pos = np.arange(len(gb_values))

	# plt.figure(figsize=(1 * len(gb_values), 4))

	width = 0.15
	bar1 = plt.bar(x_pos - width*2.5, mean['CompileTime'], width, yerr=std['CompileTime'], align='center', capsize=5)
	bar2 = plt.bar(x_pos - width*1.5, mean['KeyGenerationTime'], width, yerr=std['KeyGenerationTime'], align='center', capsize=5)
	bar3 = plt.bar(x_pos - width*0.5, mean['EncryptionTime'], width, yerr=std['EncryptionTime'], align='center', capsize=5)
	bar4 = plt.bar(x_pos + width*0.5, mean['ExecutionTime'], width, yerr=std['ExecutionTime'], align='center', capsize=5)
	bar5 = plt.bar(x_pos + width*1.5, mean['DecryptionTime'], width, yerr=std['DecryptionTime'], align='center', capsize=5)
	bar6 = plt.bar(x_pos + width*2.5, mean['ReferenceExecutionTime'], width, yerr=std['ReferenceExecutionTime'], align='center', capsize=5)
	bars = [bar1, bar2, bar3, bar4, bar5, bar6]
	bar_names = ['Derleme', 'Anahtar\nÜretimi', 'Şifreleme', 'Çalıştırma', 'Deşifreleme', 'Referans\nÇalıştırma']

	plt.xlabel('Girdi Matrisi Boyutu')
	plt.ylabel('Zaman (ms)')
	# plt.title(f'Times for Compute Distances ({title})')
	plt.xticks(x_pos, [f"{int(gb_value[0]**0.5)}x{int(gb_value[0]**0.5)}" for gb_value in gb_values], rotation = 0)
	location = (0.01,0.55)
	plt.legend(bars, bar_names)#, loc='upper left')
	plt.xlim(x_pos[0] - 0.5, x_pos[-1] + 0.5)
	# if i == 0:
	# 	plt.ylim(0,100)
	# else:
	# 	plt.ylim(bottom=0)
	plt.savefig(dir_name + distances_file.replace('.csv', '_times.png'))
	plt.clf()	# clear the saved figure

	# if i != 0:
	plt.figure(figsize=(4, 4))

	width = 0.7
	bar_mse = plt.bar(x_pos, mean['MSE'], width, yerr=std['MSE'], align='center', capsize=10, color='indianred')

	plt.xlabel('Girdi Matrisi Boyutu')
	plt.ylabel('MSE Puanı')
	# plt.title(f'MSE Scores for Compute Distances ({title})')
	plt.xticks(x_pos, [f"{int(gb_value[0]**0.5)}x{int(gb_value[0]**0.5)}" for gb_value in gb_values], rotation = 0)
	plt.xlim(x_pos[0] - 0.5, x_pos[-1] + 0.5)
	plt.ylim(bottom=0)
	plt.savefig(dir_name + distances_file.replace('.csv', '_mse.png'))