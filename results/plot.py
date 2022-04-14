import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = [8, 4]
plt.rcParams["figure.autolayout"] = True

# Header: NodeCount SimCnt CompileTime KeyGenerationTime EncryptionTime ExecutionTime DecryptionTime ReferenceExecutionTime Mse
df = pd.read_csv('results.csv').drop(columns=['SimCnt'])
gb = df.groupby(['NodeCount'])
gb_values = list(gb.groups)
mean = gb.mean()
std = gb.std()
x_pos = np.arange(len(gb_values))

print('Mean Square Errors:\n', mean['Mse'].astype(str) + u" \u00B1 " + std['Mse'].astype(str))

width = 0.15
bar1 = plt.bar(x_pos - width*2.5, mean['CompileTime'], width, yerr=std['CompileTime'], align='center', capsize=3)
bar2 = plt.bar(x_pos - width*1.5, mean['KeyGenerationTime'], width, yerr=std['KeyGenerationTime'], align='center', capsize=3)
bar3 = plt.bar(x_pos - width*0.5, mean['EncryptionTime'], width, yerr=std['EncryptionTime'], align='center', capsize=3)
bar4 = plt.bar(x_pos + width*0.5, mean['ExecutionTime'], width, yerr=std['ExecutionTime'], align='center', capsize=3)
bar5 = plt.bar(x_pos + width*1.5, mean['DecryptionTime'], width, yerr=std['DecryptionTime'], align='center', capsize=3)
bar6 = plt.bar(x_pos + width*2.5, mean['ReferenceExecutionTime'], width, yerr=std['ReferenceExecutionTime'], align='center', capsize=3)
bars = [bar1, bar2, bar3, bar4, bar5, bar6]
bar_names = ['Compile', 'Key Generation', 'Encryption', 'Execution', 'Decryption', 'Reference Execution']

plt.xlabel('Node Count')
plt.ylabel('Time')
plt.title('Benchmarking Times')
plt.xticks(x_pos, gb_values)
plt.legend(bars, bar_names, loc=(1.01,0.55))

#plt.show()
plt.savefig('benchmark_times.png')