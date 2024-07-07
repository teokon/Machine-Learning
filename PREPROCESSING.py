import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

timestamp_col = 'timestamp'

label_col = 'label'


file_paths = [
       "../Desktop/harth/S006.csv",
    "../Desktop/harth/S008.csv",
    "../Desktop/harth/S009.csv",
    "../Desktop/harth/S010.csv",
    "../Desktop/harth/S012.csv",
    "../Desktop/harth/S013.csv",
    "../Desktop/harth/S014.csv",
    "../Desktop/harth/S015.csv",
    "../Desktop/harth/S016.csv",
    "../Desktop/harth/S017.csv",
    "../Desktop/harth/S018.csv",
    "../Desktop/harth/S019.csv",
    "../Desktop/harth/S020.csv",
    "../Desktop/harth/S021.csv",
    "../Desktop/harth/S022.csv",
    "../Desktop/harth/S023.csv",
    "../Desktop/harth/S024.csv",
    "../Desktop/harth/S025.csv",
    "../Desktop/harth/S026.csv",
    "../Desktop/harth/S027.csv",
    "../Desktop/harth/S028.csv",
    "../Desktop/harth/S029.csv",
 
   
]
# Φόρτωση και ενοποίηση όλων των CSV αρχείων σε ένα DataFrame
all_data = pd.concat([pd.read_csv(file) for file in file_paths])

all_data = all_data.drop(columns=['index', 'Unnamed: 0'])
print(all_data.head())

# basic stats
basic_stats = all_data.describe()
print(basic_stats)
#sampling
sampled_data = all_data.sample(frac=0.1)  
# Δημιουργία ιστογραμμάτων για κάθε αριθμητική στήλη
for column in sampled_data.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(10, 6))
    plt.hist(sampled_data[column], bins=30, edgecolor='black')
    plt.title(f'Κατανομή της {column}')
    plt.xlabel(column)
    plt.ylabel('Συχνότητα')
    plt.show()
                                                                 #boxplot
for column in sampled_data.select_dtypes(include=['float64', 'int64']).columns:                         
 if column != label_col:  
      plt.figure(figsize=(10, 6))
      sns.boxplot(x=label_col, y=column, data=sampled_data)
      plt.title(f'Boxplot της {column} ομαδοποιημένη ανά {label_col}')
      plt.show()

x_col = 'back_x'
y_col = 'back_y'

plt.figure(figsize=(10, 6))
sns.scatterplot(x=x_col, y=y_col, data=sampled_data,hue=label_col )
plt.title(f'Scatter Plot του {x_col} έναντι του {y_col}, ομαδοποιημένο ανά {label_col}')
plt.show()

 #Δημιουργία ιστογραμμάτων για κάθε αριθμητική στήλη
for column in sampled_data.select_dtypes(include=['float64', 'int64']).columns:
     min_val = sampled_data[column].min()
     max_val = sampled_data[column].max()
    
     plt.figure(figsize=(10, 6))
     sns.histplot(data=sampled_data, x=column, hue=label_col, multiple='stack', palette="deep", binrange=(min_val, max_val))
     plt.title(f'Ιστόγραμμα της {column} ομαδοποιημένο ανά label')
     plt.xlabel(column)
     plt.ylabel('Συχνότητα')
     plt.show()



     # Επιλογή μόνο των αριθμητικών στηλών

columns = sampled_data.select_dtypes(include=['float64', 'int64']).columns

# Δημιουργία scatter plots για κάθε ζευγάρι αριθμητικών στηλών με ομαδοποίηση ανά label
for i in range(len(columns)):
    for j in range(i + 1, len(columns)):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=sampled_data, x=columns[i], y=columns[j],  palette='viridis')
        plt.title(f'Scatter Plot: {columns[i]} vs {columns[j]}')
        plt.xlabel(columns[i])
        plt.ylabel(columns[j])
        plt.show()
        


numeric_data = sampled_data.select_dtypes(include=['float64'])
corr = numeric_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()




