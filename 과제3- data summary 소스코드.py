import pandas as pd
import numpy as np

xlfile = 'db_score.xlsx'

df = pd.read_excel(xlfile)
#print(df)


mean = dict()
for col in df.columns.values:
    if ( col == 'grade' or col == 'sno' ):
        continue
    mean[col] = df[col].mean()
print("\nmean:")
print(mean)



median = dict()
for col in df.columns.values:
    if ( col == 'grade' or col == 'sno' ):
        continue
    median[col] = df[col].median()
print("\nmedian:")
print(median)


grade, count = np.unique(df['grade'], return_counts=True)

max_count = 1

mode = []

for i in range(len(grade)):
    if count[i] > max_count:
        mode = []
        mode.append(grade[i])
        max_count = count[i]
    elif count[i] == max_count:
        mode.append(grade[i])
 
print("\nmode:")       
print(mode)



std = dict()
var = dict()
aad = dict()
mad = dict()

for col in df.columns.values:
    if ( col == 'grade' or col == 'sno' ):
        continue
    std[col] = df[col].std()
    var[col] = df[col].var()
print("\nstd:")
print(std)

print("\nvar:")
print(var)



for col in df.columns.values:
    if ( col == 'grade' or col == 'sno' ):
        continue
    aad[col] = np.mean(abs(df[col] - mean[col]))
    mad[col] = np.median(abs(df[col] - mean[col]))
    
print("\naad:")
print(aad)    

print("\nmad:")
print(mad) 


import matplotlib.pyplot as plt


p = [x for x in range(0,101,10)]

for col in df.columns.values:
    if ( col == 'grade' or col == 'sno' ):
        continue
    percentile = np.percentile(df[col], p)
    plt.plot(p, percentile, 'o-')
    plt.xlabel('percentile')
    plt.ylabel(col)
    plt.xticks(p)
    plt.yticks(np.arange(0, max(percentile)+1, max(percentile)/10.0))
    plt.show()



#plt.figure(figsize=(12, 10))
boxplot = df[['attendance', 'homework', 'discussion', 'midterm', 'final', 'score'  ]].boxplot()
plt.show()


print('\nhistogram:')
for col in df.columns.values:
    if ( col == 'grade' or col == 'sno' ):
        continue    
    plt.hist(df[col], facecolor='blue', bins=20)
    plt.xlabel(col)
    plt.show()



print('\nscatter plot:')
for col1 in df.columns.values:
    if ( col1 == 'grade' or col1 == 'sno' ):
        continue   
    for col2 in df.columns.values:
        if ( col2 == 'grade' or col2 == 'sno' or col1==col2  or col1 < col2):
            continue   
        plt.scatter(df[col1], df[col2])
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.show()
        


import seaborn as sns        
sns.pairplot(df[['attendance', 'homework', 'discussion', 'midterm', 'final', 'score'  ]])



