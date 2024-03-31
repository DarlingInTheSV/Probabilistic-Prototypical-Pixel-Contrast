
import torch
def label_onehot(inputs, num_class):
    '''
    inputs is class label
    return one_hot label
    dim will be increasee
    '''
    batch_size, _, image_h, image_w = inputs.shape
    inputs = torch.relu(inputs)
    outputs = torch.zeros([batch_size, num_class, image_h, image_w]).to(inputs.device)
    tmp = outputs.scatter_(1, inputs, 1.0)
    return tmp