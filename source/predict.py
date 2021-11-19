'''
This script generates predictions for a set of pairs given a trained neural network.
'''
import gzip,random,numpy as np,argparse

# Pytorch
import torch
from torch.autograd import Variable
import torch.utils.data
from shared import *

def predict(net,
            human_test_filename,mouse_test_filename,output_filename,
            test_data_size,batch_size,
            num_human_features,num_mouse_features,
            human_rnaseq_range,mouse_rnaseq_range):

    # Difference between maximum and minimum RNA-seq signals
    hrr = human_rnaseq_range[1]-human_rnaseq_range[0]
    mrr = mouse_rnaseq_range[1]-mouse_rnaseq_range[0]

    # Make predictions and write to output
    with gzip.open(human_test_filename,'rb') as hf,\
            gzip.open(mouse_test_filename,'rb') as mf,\
            gzip.open(output_filename if output_filename.endswith('.gz') else output_filename+'.gz','wb') as fout:

        for i in range(int(test_data_size/batch_size)+1): # iterate through each batch
            current_batch_size = batch_size if i<int(test_data_size/batch_size) else test_data_size%batch_size

            if current_batch_size==0:
                break

            X = np.zeros((current_batch_size,num_human_features),dtype=float) # to store human data
            Y = np.zeros((current_batch_size,num_mouse_features),dtype=float) # to store mouse data

            for j in range(current_batch_size): # iterate through each sample within the batch
                hl = hf.readline().strip().split(b'|')
                ml = mf.readline().strip().split(b'|')

                # Indices of the non-zero feature indices
                nonzero_human_feature_indices = [int(s) for s in hl[1].strip().split()]
                nonzero_mouse_feature_indices = [int(s) for s in ml[1].strip().split()]

                # Normalize RNA-seq values
                real_valued_human_features = [(float(s)-human_rnaseq_range[0])/hrr
                                              for s in hl[2].strip().split()] if len(hl)>1 else []
                real_valued_mouse_features = [(float(s)-mouse_rnaseq_range[0])/mrr
                                              for s in ml[2].strip().split()] if len(ml)>1 else []

                # Set non-zero features to the corresponding values
                num_nonzero_binary_human_features = len(nonzero_human_feature_indices)-len(real_valued_human_features)
                h = np.concatenate((np.ones((num_nonzero_binary_human_features)),real_valued_human_features))
                X[j, nonzero_human_feature_indices] = h
                num_nonzero_binary_mouse_features = len(nonzero_mouse_feature_indices)-len(real_valued_mouse_features)
                m = np.concatenate((np.ones((num_nonzero_binary_mouse_features)),real_valued_mouse_features))
                Y[j, nonzero_mouse_feature_indices] = m

            # Convert feature matrices for PyTorch
            X = Variable(torch.from_numpy(X).float())
            Y = Variable(torch.from_numpy(Y).float())
            inputs = torch.cat((X,Y),1) # concatenate human and mouse data

            # Make prediction on current batch
            y_pred = net(inputs) # put the feature matrix into the provided trained PSNN
            y_pred = y_pred.data

            # Write predicted probabilities of the current batch
            sample_output = [str(round(y_pred[j],7)) for j in range(current_batch_size)]
            l = '\n'.join(sample_output)+'\n'
            fout.write(l.encode())


def main():
    epilog = '# Example: python source/predict.py -t NN/odd_ensemble/NN_1_*.pt -H data/even_all.h.gz -M \
    data/even_all.m.gz -d 16627449 -o NN/odd_ensemble/even_all_1.gz'

    parser = argparse.ArgumentParser(prog='python source/predict.py',
                                     description='Generate predictions given a trained neural network',
                                     epilog=epilog)
    parser.add_argument('-s', '--seed', help='random seed (default: 1)', type=int, default=1)
    parser.add_argument('-b', '--batch-size', help='batch size (default: 128)', type=int, default=128)

    g1 = parser.add_argument_group('required arguments specifying input and output')
    g1.add_argument('-t', '--trained-classifier-filename', required=True, help='path to a trained classifier (.pt)',
                    type=str)
    g1.add_argument('-H', '--human-feature-filename', required=True, help='path to human feature data file', type=str)
    g1.add_argument('-M', '--mouse-feature-filename', required=True, help='path to mouse feature data file', type=str)
    g1.add_argument('-d', '--data-size', required=True, help='number of samples', type=int)
    g1.add_argument('-o', '--output-filename', required=True, help='path to output file', type=str)

    g1.add_argument('-hf', '--num-human-features',
                    help='number of human features in input vector (default: 8824)', type=int, default=8824)
    g1.add_argument('-mf', '--num-mouse-features',
                    help='number of mouse features in input vector (default: 3113)', type=int, default=3113)
    g1.add_argument('-hrmin', '--human-rnaseq-min',
                    help='minimum expression level in human RNA-seq data (default: 8e-05)', type=float, default=8e-05)
    g1.add_argument('-hrmax', '--human-rnaseq-max',
                    help='maximum expression level in human RNA-seq data (default: 1.11729e06)', type=float, default=1.11729e06)
    g1.add_argument('-mrmin', '--mouse-rnaseq-min',
                    help='minimum expression level in mouse RNA-seq data (default: 0.00013)', type=float, default=0.00013)
    g1.add_argument('-mrmax', '--mouse-rnaseq-max',
                    help='maximum expression level in mouse RNA-seq data (default: 41195.3)', type=float, default=41195.3)

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load previously trained classifier
    net = torch.load(args.trained_classifier_filename)
    net.eval() # make sure it's in evaluation mode

    # Make predictions
    predict(net,
            args.human_feature_filename,args.mouse_feature_filename,args.output_filename,
            args.data_size,args.batch_size,
            args.num_human_features,args.num_mouse_features,
            [args.human_rnaseq_min,args.human_rnaseq_max],
            [args.mouse_rnaseq_min,args.mouse_rnaseq_max])

if __name__ == "__main__":
    main()
