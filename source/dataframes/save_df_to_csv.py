import pandas as pd
import Grasshopper
import ghpythonlib.components as ghcomp

def save_dataframe_to_csv(df, file_path):
    # Save DataFrame to CSV
    df.to_csv(file_path, index=False)  # index=False avoids writing row numbers
    return "File saved successfully"

# Inputs
# df = x  # The DataFrame to be saved
# file_path = y  # The path where the CSV will be saved
# button = z  # The button state

# Check if button is pressed
if save_button:
    message = save_dataframe_to_csv(df, file_path)
# else:
#     message = "Press the button to save the file"

# # Output
# a = message  # Output a success message or status

