#! python3
#r: openpyxl, psycopg2, Ipython

#THIS IS NECESSARY TO LOAD PANDAS otherwise it bugs
import locale
locale.setlocale(locale.LC_ALL, 'en_US')

############ NOTES ####################################################
## Works but intput has to be list so data from merge component has to be flattened. 
## Output should be data tree instead

import pandas as pd

# Initialize with the first DataFrame
combined_df = df_list[0]

# Join each subsequent DataFrame
for df in df_list[1:]:
    combined_df = combined_df.join(df, how='outer')  # or use 'inner' if you want only matching keys

# Result DataFrame
merged_df = combined_df
