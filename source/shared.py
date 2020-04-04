'''
This script defines the neural network architecture and functions useful in training
'''

import sys,scipy.sparse,numpy as np,torch

# Print progress given a percentage and appropriate message
def printProgress(p,messsage):
    sys.stdout.write("\r[%s%s] %d%% %s    " % ("=" * p, " " * (100 - p), p, messsage))
    sys.stdout.flush()

def readBatch(files_to_read,batch_size,lines_to_read,num_features,rnaseq_range):

    # File pointers
    positive_human_data_file = files_to_read[0]
    positive_mouse_data_file = files_to_read[1]
    negative_human_data_file = files_to_read[2]
    negative_mouse_data_file = files_to_read[3]

    # True if specific lines are given to read, false if all the lines should be read
    read_specific_lines = isinstance(lines_to_read,np.ndarray)

    # Store RNA-seq ranges for normalization
    human_rnaseq_range,mouse_rnaseq_range = rnaseq_range[0],rnaseq_range[1]

    # Difference between maximum and minimum RNA-seq signal values
    hrr,mrr = human_rnaseq_range[1]-human_rnaseq_range[0], mouse_rnaseq_range[1]-mouse_rnaseq_range[0]

    # Lists needed to construct a SciPy sparse matrix
    row,col,data = [],[],[] # row indices, column indices, feature values

    i = 0 # example index
    while i<batch_size:
        j = int(i/2.) # line offset array index

        if read_specific_lines:
            positive_human_data_file.seek(lines_to_read[0,j])
            positive_mouse_data_file.seek(lines_to_read[1,j])
            negative_human_data_file.seek(lines_to_read[2,j])
            negative_mouse_data_file.seek(lines_to_read[3,j])

        ### Read one positive example
        hl = positive_human_data_file.readline().strip().split(b'|')
        ml = positive_mouse_data_file.readline().strip().split(b'|')
        positive_nonzero_human_feature_indices = [int(s) for s in hl[1].strip().split()]
        positive_nonzero_mouse_feature_indices = [int(s)+num_features[0] for s in ml[1].strip().split()]

        # Normalize RNA-seq values for the positive example
        positive_real_valued_human_features = [(float(s)-human_rnaseq_range[0])/hrr
                                               for s in hl[2].strip().split()] if len(hl)>1 else []
        positive_real_valued_mouse_features = [(float(s)-mouse_rnaseq_range[0])/mrr
                                               for s in ml[2].strip().split()] if len(ml)>1 else []

        ### Read one negative example
        hl = negative_human_data_file.readline().strip().split(b'|')
        ml = negative_mouse_data_file.readline().strip().split(b'|')
        negative_nonzero_human_feature_indices = [int(s) for s in hl[1].strip().split()]
        negative_nonzero_mouse_feature_indices = [int(s)+num_features[0] for s in ml[1].strip().split()]

        # Normalize RNA-seq values for the negative example
        negative_real_valued_human_features = [(float(s)-human_rnaseq_range[0])/hrr
                                               for s in hl[2].strip().split()] if len(hl)>1 else []
        negative_real_valued_mouse_features = [(float(s)-mouse_rnaseq_range[0])/mrr
                                               for s in ml[2].strip().split()] if len(ml)>1 else []

        ### Save data for the two examples
        row += [i]*(len(positive_nonzero_human_feature_indices)
                    +len(positive_nonzero_mouse_feature_indices))\
               +[i+1]*(len(negative_nonzero_human_feature_indices)
                       +len(negative_nonzero_mouse_feature_indices))
        col += positive_nonzero_human_feature_indices\
               +positive_nonzero_mouse_feature_indices\
               +negative_nonzero_human_feature_indices\
               +negative_nonzero_mouse_feature_indices
        data += [1]*(len(positive_nonzero_human_feature_indices)
                     -len(positive_real_valued_human_features))\
                +positive_real_valued_human_features\
                +[1]*(len(positive_nonzero_mouse_feature_indices)
                      -len(positive_real_valued_mouse_features))\
                +positive_real_valued_mouse_features\
                +[1]*(len(negative_nonzero_human_feature_indices)
                      -len(negative_real_valued_human_features))+\
                negative_real_valued_human_features\
                +[1]*(len(negative_nonzero_mouse_feature_indices)
                      -len(negative_real_valued_mouse_features))\
                +negative_real_valued_mouse_features

        i += 2 # read two examples in the while loop, one positive example and one negative example

    # Build a SciPy sparse matrix with feature data and convert it into an array
    X = scipy.sparse.coo_matrix((data,(row,col)),shape=(batch_size,num_features[0]+num_features[1])).toarray()

    # Build a label array
    Y = np.zeros(batch_size,dtype=int) # label
    Y[::2] = 1 # odd examples are positive examples

    # Convert data for PyTorch
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()
    X,Y = torch.autograd.Variable(X), torch.autograd.Variable(Y)
    return X,Y

# A pseudo-Siamese neural network
class PseudoSiameseNet(torch.nn.Module):
    def __init__(self,num_human_features,num_mouse_features,num_layers,num_neurons,dropout_rate):
        """
        num_layer_species: number of layers in species-specific sub-networks
        num_layer_final: number of layers in final sub-network
        num_neuron_human: number of neurons in each layer of human-specific sub-network
        num_neuron_mouse: number of neurons in each layer of mouse-specific sub-network
        num_neuron_final: number of neurons in each layer of final sub-network
        """
        super(PseudoSiameseNet, self).__init__()
        self.num_human_features = num_human_features
        self.num_layer_species = num_layers[0]
        self.num_layer_final = num_layers[2]
        self.num_neuron_human = num_neurons[:self.num_layer_species]
        self.num_neuron_mouse = num_neurons[2:2+self.num_layer_species]
        self.num_neuron_final = num_neurons[4:4+self.num_layer_final]

        # Sequence of operations done only on either human or mouse features
        self.human_layers = torch.nn.Sequential()
        self.mouse_layers = torch.nn.Sequential()
        for i in range(self.num_layer_species):
            if i==0: # from input features to first hidden layer
                self.human_layers.add_module('h'+str(i),
                                             torch.nn.Linear(num_human_features,int(self.num_neuron_human[i]),bias=False))
                self.mouse_layers.add_module('m'+str(i),
                                             torch.nn.Linear(num_mouse_features,int(self.num_neuron_mouse[i]),bias=False))
            else:
                self.human_layers.add_module('h'+str(i),
                                             torch.nn.Linear(int(self.num_neuron_human[i-1]),int(self.num_neuron_human[i])))
                self.mouse_layers.add_module('m'+str(i),
                                             torch.nn.Linear(int(self.num_neuron_mouse[i-1]),int(self.num_neuron_mouse[i])))
            self.human_layers.add_module('h'+str(i)+'dropout',torch.nn.Dropout(p=dropout_rate)) # dropout
            self.human_layers.add_module('hr'+str(i),torch.nn.ReLU()) # relu
            self.mouse_layers.add_module('m'+str(i)+'dropout',torch.nn.Dropout(p=dropout_rate)) # dropout
            self.mouse_layers.add_module('mr'+str(i),torch.nn.ReLU()) # relu

        # Sequence of operations done on concatenated output of species-specific sub-networks
        self.final_layers = torch.nn.Sequential()
        for i in range(self.num_layer_final):
            if i==0: # from concatenated output of species-specific sub-networks to the first layer of final sub-network
                self.final_layers.add_module('c'+str(i),
                                             torch.nn.Linear(int(self.num_neuron_human[-1])+int(self.num_neuron_mouse[-1]),int(self.num_neuron_final[i])))
            else:
                self.final_layers.add_module('c'+str(i),
                                             torch.nn.Linear(int(self.num_neuron_final[i-1]),int(self.num_neuron_final[i])))
            self.final_layers.add_module('cd'+str(i),torch.nn.Dropout(p=dropout_rate))
            self.final_layers.add_module('cr'+str(i),torch.nn.ReLU())

        # Output layer
        self.final_layers.add_module('end',torch.nn.Linear(int(self.num_neuron_final[-1]),1)) # from last layer to single output
        self.final_layers.add_module('sigmoid',torch.nn.Sigmoid())

    def forward(self,x):
        h = self.human_layers.forward(x[:,:self.num_human_features]) # human-specific sub-network
        m = self.mouse_layers.forward(x[:,self.num_human_features:]) # mouse-specific sub-network
        c = torch.cat((h,m),1) # concatenate the output from human-specific sub-network and mouse-specific sub-network
        y = self.final_layers.forward(c) # final/final sub-network
        y = y.view(c.size()[0])
        return y