import pandas as pd

# Step 1: Read the CSV file into a DataFrame
df = pd.read_csv('results_multiclass.csv')

# Step 2: Remove the last two columns from the DataFrame
df = df.iloc[:, :-2]

# Step 3: Write the modified DataFrame to a new CSV file
df.to_csv('results_multiclass_2.csv', index=False)

print("The last two columns have been removed and the new file 'results_multiclass_2.csv' has been created.")
