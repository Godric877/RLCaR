from trace_loader import load_traces
import pandas as pd
import matplotlib.pyplot as plt

df = load_traces("test", 128, 0)
print(df[1].nunique())
#fig, ax = plt.plot()
x = df[1].value_counts()
print(x[x>1])
#plt.show()