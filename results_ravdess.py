from experiment.stats import get_results
import matplotlib.pyplot as plt 
import numpy as np 

ps, ws = get_results("clean_logs/iemocap_f1acc_k5_1637213724.json")


bins = np.arange(0,1,0.05)
print(bins)
print(ps)

fig, ax = plt.subplots()
ax.hist(ps,bins=bins)
ax.set_xticks(bins)
plt.show()