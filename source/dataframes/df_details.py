info = dataFrame.info

head_x10 = dataFrame.head(10)
tail_x10 = dataFrame.tail(10)
shape = dataFrame.shape

#Add up all null values
number_of_null_values = dataFrame.isnull().sum()



# columns = dataFrame.columns
# Get the columns as a list
column_name = dataFrame.columns.tolist()

# Output the list
# print(column_name)