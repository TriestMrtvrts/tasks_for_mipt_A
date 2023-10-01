import pandas as pd

df1 = pd.read_csv('mars-private_test-reg.csv')
df2 = pd.read_csv('pred_ens.csv')

mdf = pd.concat([df1, df2], axis=1)
mdf.to_csv('merged_reg.csv', index=False)
