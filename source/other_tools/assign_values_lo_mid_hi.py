# Ensure num_list is a list, even if a single number is provided
if isinstance(num_list, int) or isinstance(num_list, float):
    num_list = [num_list]

# Initialize the output lists
below = []
in_between = []
above = []

# Iterate through the list and classify numbers
for num in num_list:
    if num < loVal:
        below.append(num)
    elif loVal <= num <= hiVal:
        in_between.append(num)
    else:
        above.append(num)

# # Assign to Grasshopper outputs
low = below
medium = in_between
high = above
