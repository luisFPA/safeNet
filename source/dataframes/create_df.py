import pandas as pd

def tree_to_dataframe(columns, data_tree):
    # Ensure the number of branches matches the number of columns
    if len(columns) != len(data_tree.Branches):
        raise ValueError("The number of columns must match the number of branches in the data tree")
    
    # Convert each branch in the data tree to a column
    data_dict = {col: data_tree.Branches[i] for i, col in enumerate(columns)}
    
    # Create DataFrame
    df = pd.DataFrame(data_dict)
    return df

# Inputs
column_list = columns  # List of column names
data_tree = data  # Data should be a tree joined with entwined

# Convert the data tree to a DataFrame
df = tree_to_dataframe(columns, data_tree)

# Output
dataframe = df  # Output the DataFrame