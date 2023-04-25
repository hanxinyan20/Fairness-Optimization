import matplotlib.pyplot as plt
import numpy as np
import torch
loaded_data = np.load("binarized_purged_querynorm.npz", allow_pickle=True)
print(loaded_data['format_version'])
feature_map = loaded_data['feature_map'].item()
train_feature_matrix = loaded_data['train_feature_matrix']
train_doclist_ranges = loaded_data['train_doclist_ranges']
train_label_vector   = loaded_data['train_label_vector']
valid_feature_matrix = loaded_data['valid_feature_matrix']
valid_doclist_ranges = loaded_data['valid_doclist_ranges']
valid_label_vector   = loaded_data['valid_label_vector']
test_feature_matrix  = loaded_data['test_feature_matrix']
test_doclist_ranges  = loaded_data['test_doclist_ranges']
test_label_vector    = loaded_data['test_label_vector']
print("train_label_vector:",train_label_vector.type)
# x=torch.nn.functional.softmax(torch.Tensor([0,1,2]))
# print(x)
# x = torch.tensor([[1,2,3],[5,6,7]])
# print(x.data[0][1].type())