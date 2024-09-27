#! python3
#r: openpyxl, psycopg2, Ipython

import matplotlib.pyplot as plt
import matplotlib

# Ensure you have the correct imports
import matplotlib.pyplot as plt

def plot_graph(n_epochs, train_accs, valid_accs, plot):
    if not plot:
        return  # If plot is False, exit the function

    # Clear the current figure to avoid plotting over previous plots
    plt.clf()
    plt.close()
    
    # Plot training and validation accuracy
    plt.plot(range(n_epochs), train_accs, label='Training Accuracy')
    plt.plot(range(n_epochs), valid_accs, label='Validation Accuracy')

    # Add title and labels
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    # Display legend
    plt.legend(loc='best')

    # Display the plot
    plt.show()

# Call the plot function
# if plot:
plot_graph(n_epochs, training_acc, validation_acc, plot)


######### OUTPUTS ###########################################
max_train_acc = max(training_acc)
max_val_acc = max(validation_acc)
