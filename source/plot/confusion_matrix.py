#! python3
#r: openpyxl, psycopg2, Ipython

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score, accuracy_score
import numpy as np

# Define the plot function
def plot_graph(x, y):
    # Clear the figure to avoid duplicating plots
    plt.clf()
    plt.close()

    # Generate confusion matrix and plot
    cm = confusion_matrix(y, x, labels=np.unique(x))  # Ensure labels are set
    print(np.unique(true_label))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(x))
    disp.plot()

    # Display the plot
    plt.show()





# Check if the plot should be made and ensure it's done only once
if plot:
    plot_graph(predicted_label, true_label)
else:
    print("No valid data to plot.")

# Calculate metrics with proper handling for multi-class classification
accuracy_sc = accuracy_score(true_label, predicted_label)
precision_sc = precision_score(true_label, predicted_label, average='weighted')  # Use 'weighted' for multi-class
recall_sc = recall_score(true_label, predicted_label, average='weighted')        # Use 'weighted' for multi-class
f1_sc = f1_score(true_label, predicted_label, average='weighted')                # Use 'weighted' for multi-class

print("Accuracy:", accuracy_sc)
print("Precision:", precision_sc)
print("Recall:", recall_sc)
print("F1 Score:", f1_sc)