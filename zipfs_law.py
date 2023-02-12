import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# =============================================================================
# Zipf's law is a model about the frequency distribution of words in a language 
# It is expressed  as a power-law relationship between the frequency of an element f and its rank r
# c is a constant and k is the power-law exponent
# =============================================================================

zipfs_law = lambda r, c, k: c*(r**-k) 

r = np.arange(1,51,1) # 1 to 50 ranking
n_words = 0 # number of words

files = [f for f in os.listdir('txts') if f.endswith('.txt')] #search all files that endswith .txt
d = {} 

for file in files:
    with open(f'txts/{file}', 'r') as f:
        text = f.read() #read text
        text = re.sub(r'[^a-zA-Z0-9]',' ', text) # remove non-alphanumeric characters
        text = text.lower().rstrip() # lower case + remove any white spaces at the end
        wds = text.split() # split a string into a list
        n_words += len(wds)
        if len(wds)<1000: # check if the file contains less than 100 words
            print(f'DISCARDED FILE! {file} contains less than 1000 words')
            continue
        
        for w in wds: #words counter
            d[w] = d.get(w,0)+1

        sort_w = dict(sorted(d.items(), key=lambda x: x[1], reverse=True)) # sort the dictionary in a descending order
        sort_w = list(sort_w.items())[0: 50] # get  the 50 most common words 

mst_cmmn_wrds_df = pd.DataFrame({'Word':list(zip(*sort_w))[0], 'N_times':list(zip(*sort_w))[1]}) # data frame of the 50 most common words  

mst_cmmn_wrds_df.index = range(1,51)
mst_cmmn_wrds_df['Pr[%]'] = mst_cmmn_wrds_df['N_times']/n_words*100
mst_cmmn_wrds_df['Word'] = [f.upper() for f in mst_cmmn_wrds_df['Word']]
print(mst_cmmn_wrds_df)

c, cov = curve_fit(zipfs_law,r,mst_cmmn_wrds_df['Pr[%]'])
R2 = r2_score(mst_cmmn_wrds_df['Pr[%]'],zipfs_law(r, c[0], c[1]))
print(f'R2: {R2:.2%}')

f, ax = plt.subplots(figsize=(10,12),facecolor='.85')
ax.bar(mst_cmmn_wrds_df['Word'], mst_cmmn_wrds_df['Pr[%]'], alpha=.3)
ax.plot(mst_cmmn_wrds_df.index-1, zipfs_law(r, c[0], c[1]), linewidth=4, color='k')
ax.set_ylim([0,6])
plt.xticks(fontsize=8,rotation=90)
ax.set_xlabel('Words', fontsize=15)
ax.set_ylabel('Probability [%]', fontsize=15)
plt.title(f"Zipf's law | Nº of words used: {n_words}", fontsize=16)
plt.show()
plt.savefig('zipfs_law_image.png')
