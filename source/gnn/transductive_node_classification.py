#! python3
#r: openpyxl, psycopg2, Ipython

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv
from sklearn.model_selection import train_test_split
from collections import Counter


# import sys  # To use sys.exit()
# import numpy as np

# print(graph.ndata["classes"])



######### NEEDED TOOLS ############################################################
# Create a dict keyed by the dataframe original order and valued by the dgl index
reorder_dict = {}
# Iterate over the DataFrame's index and populate the dictionary
for i in range(len(graph.ndata['df_order'])):
    # Extract the value at index i
    key = graph.ndata['df_order'][i].item()  # Convert tensor to a scalar
    reorder_dict[key] = i
# print(reorder_dict)







######### PREPARE GRAPH FOR TRANSDUCTIVE CLASSIFICATION TASK #######################
# `graph` is a pre-existing DGL graph object with node data from another component
# and that `graph.ndata["classes"]` is a tensor containing the class labels for the nodes.



# Get labeled and unlabeled nodes
labeled_nodes = [i.item() for i, j in zip(graph.nodes(), graph.ndata["classes"]) if j.item() != class_to_predict]
un_labeled_nodes = [i.item() for i, j in zip(graph.nodes(), graph.ndata["classes"]) if j.item() == class_to_predict]


# print(labeled_nodes)
# print(un_labeled_nodes)
print("labeled nodes: ", len(labeled_nodes))
print("unlabeled nodes: ", len(un_labeled_nodes))

## Separating between training and others
train_nodes, test_nodes = train_test_split(labeled_nodes, random_state=1, test_size=0.1) #Test
train_nodes, validation_nodes = train_test_split(train_nodes, random_state=1, test_size=0.11) #Validation

prediction_nodes = un_labeled_nodes ###PMASK
# print(len(train_nodes))
# print(len(test_nodes))
# print(len(validation_nodes))
# print(len(prediction_nodes))

## Create train/validation/test/prediction masks
# Each mask is just a boolean array of the same size of the number of nodes, positions of nodes to be included are set to True, and the rest is False
num_nodes = graph.number_of_nodes()
# print(num_nodes)



## METHOD 1 (ORIGINAL)
# Train mask
# print(sorted(train_nodes))
train_mask={i.item() :False for i in graph.nodes()}
# print(train_mask)
[train_mask.update({n:True}) for n in train_nodes]
# print(train_mask)

# Validation mask
# print(sorted(validation_nodes))
validation_mask={i.item() :False for i in graph.nodes()}
[validation_mask.update({n:True}) for n in validation_nodes]
# print(validation_mask)

# Test mask
# print(sorted(test_nodes))
test_mask={i.item() :False for i in graph.nodes()}
[test_mask.update({n:True}) for n in test_nodes]
# print(test_mask)

# Prediction mask
# print(sorted(prediction_nodes))
prediction_mask={i.item() :False for i in graph.nodes()} #########PMASK
[prediction_mask.update({n:True}) for n in prediction_nodes] #######PMASK
# print(prediction_mask)


## Assign mask to tensor
graph.ndata["train_mask"]=torch.tensor(list(train_mask.values()))
graph.ndata["validation_mask"]=torch.tensor(list(validation_mask.values()))
graph.ndata["test_mask"]=torch.tensor(list(test_mask.values()))
graph.ndata["prediction_mask"]=torch.tensor(list(prediction_mask.values())) #########PMASK




######### LEARNING #######################
## Training of model based on unknown classes
## GNN module
class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = SAGEConv(in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h

## Get the features and label sizes
n_features = graph.ndata["feat"].shape[1]
classes_list = [t[0] for t in graph.ndata["classes"].tolist()]
n_labels = len(Counter(classes_list).keys())

# print(n_features)
# print(classes_list)
print("nLabels", n_labels)
# print(Counter(classes_list).keys())


## Hyperparameters 
n_hidden = neurons  # Example value, adjust as needed...default 4
lr = learning_rate     # Example value, adjust as needed...default 0.01
n_epochs = epochs # Example value, adjust as needed...default 70
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # cpu or cuda. Current script looks for cuda first. Uses cpu if not available


# print(graph.ndata['feat'])


# Actual training
if training: #Looks for a button from the component called "training"
    
    model = SAGE(n_features, n_hidden, n_labels)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model_path = "best_model"
    torch.manual_seed(40)
    best_val_acc = 0
    best_test_acc = 0

    features = graph.ndata['feat']
    labels = graph.ndata['classes']  
    train_mask_1 = graph.ndata['train_mask']
    val_mask_1 = graph.ndata['validation_mask']
    test_mask_1 = graph.ndata['test_mask']

    ## Test values
    # print(features)
    # print(labels)
    # print(train_mask_1)
    # print(val_mask_1)
    # print(test_mask_1)


    train_accs = []
    valid_accs = []


    for e in range(n_epochs):
        model.train() ## Test this
        optimizer.zero_grad() ## Test this

        # Forward pass
        logits = model(graph, features)

        # Compute predictions
        pred = logits.argmax(1)
        # print(pred)

        # Compute loss
        # loss = F.cross_entropy(logits_1d, labels_1d)
        loss = F.cross_entropy(logits[train_mask_1], labels[train_mask_1].squeeze())
        # loss = nn.CrossEntropyLoss()

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask_1] == labels).float().mean()
        val_acc = (pred[val_mask_1] == labels).float().mean()
        test_acc = (pred[test_mask_1] == labels[test_mask_1]).float().mean()




        train_accs.append(train_acc)
        valid_accs.append(val_acc)
        # print(val_acc)

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # print(f'Epoch {e}, Loss: {loss.item():.3f}, Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}')

    # Load the best model
    # model.load_state_dict(torch.load(model_path)) ## Test this...hide if not used
# print(train_accs)
# print(valid_accs)




######### EVALUATION AND COMPLETING MISSING CLASSES #######################
# This section would involve using the trained model to predict labels for the nodes in `prediction_mask`.

## Test ground vs prediction (Check this for errors since unknown class is showing up!!!!!!!!)
#There should be no predictions of the unknown class since it is never seen in training
pred_values = [i.item() for i in pred] #Counts predictions across all data points (train, validation, and test).
print(Counter(pred_values))
# print(pred_values)

##Complete list of predicted values from pred_values reordered according to original dataframe
pred_values_reordered = [pred_values[reorder_dict[i]] for i in sorted(reorder_dict.keys())] # Reorder pred_values according to original dataframe to match nodes
# print(pred_values_reordered)

test_values = [i.item() for i in pred[test_mask_1]] #Counts predictions only for the test data points
print(Counter(test_values))
# print(test_values)



y_test = labels[test_mask_1]
predictions = pred[test_mask_1]
# print(y_test)
# print(predictions)
# print(pred[test_mask_1] )



## Generate values based on predicted and actual classes, using a custom logic that considers a prediction_mask.
classes_list=[t[0] for t in graph.ndata["classes"].tolist()]
classes_list_reordered=[classes_list[reorder_dict[i]] for i in sorted(reorder_dict.keys())]
# Initialize a list to store the assigned values

assigned_values = [] # List consists of a combo of actual values for what we know and newly predicted values
# Iterate over nodes and their corresponding predicted and actual classes
for i, (pred_value, actual_class) in enumerate(zip(pred_values_reordered, classes_list_reordered)):
    # Check if the prediction was correct
    if prediction_mask[reorder_dict[i]]:  # True if prediction is correct
        assigned_value = pred_value  # Assign the predicted class value
    else:  # False if prediction is incorrect
        assigned_value = actual_class  # Assign the actual class value
    
    # Append the assigned value to the list
    assigned_values.append(assigned_value)
# print(assigned_values


##OUTPUTS ##############################################################  

## Main predicted lists
pred_only = pred_values_reordered
act_pred = assigned_values




## For traning vs validation accuracy chart
training_acc = [t.item() for t in train_accs]
validation_acc = [t.item() for t in valid_accs]
# print (training_acc)
# print (validation_acc)




## For confusion matrix
## flatten to output
predicted_label = predictions.squeeze().tolist()
true_label = y_test.squeeze().tolist()

# print(predicted_label)
# print(true_label)

# print(labels)


## Complete missing classes
# print(Counter([t[0] for t in graph.ndata["classes"].tolist()]))

# print(graph.ndata["classes"].tolist())