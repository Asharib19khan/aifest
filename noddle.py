import pandas as pd

train_file = 'train.csv'  
test_file = 'test.csv'    
target_name = 'Outcome'   

df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)

print("Columns in your data:", df_train.columns.tolist())
print(f"Your target is: {target_name}")

df_train.head()