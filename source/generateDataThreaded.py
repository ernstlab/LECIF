'''
This script aggregates processed feature data into one file for a given set of genomic regions within one species.
Note that this script involves multithreading of 4 threads.
The output file has the following format:
- Each line: chr start end pos_index | <list of all active/non-zero feature indices> | <list of real values>
- Values in each list are separated by tab
- Real values correspond to the last active/non-zero feature indices. For example, if there are n real values, they
correspond to the n last active/non-zero feature indices. If there is nothing written after the second vertical bar (|),
there is no feature with real values. All active/non-zero features are binary in this case.
'''

import sys,gzip,threading,numpy as np,pandas as pd,argparse,os
from collections import defaultdict

# SOURCE: https://www.tutorialspoint.com/python3/python_multithreading.htm
class myThread (threading.Thread):
    feature_str = {key: str() for key in range(1,5)}
    print_init = 0
    
    def __init__(self, threadID, name, feature, directory, input_regions, active_indices,
                 chrom_states, cage_experiments, feature_indices):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.feature = feature
        self.directory = directory
        self.input_regions = input_regions
        self.active_indices = active_indices # indices of non-zero features
        self.chrom_states = chrom_states # number of chromatin states
        self.cage_experiments = cage_experiments # number of CAGE experiments
        self.feat_index = feature_indices[feature] # starting feature index of the thread
        self.real_values = None

    def run(self):
        if self.feature == 1:
            readDnaseChipFeature(self.directory, self.input_regions, self.active_indices)
        elif self.feature == 2:
            readChromHmmFeature(self.directory, self.input_regions, self.active_indices,
                                self.chrom_states, self.feat_index)
        elif self.feature == 3:
            readCageFeature(self.directory, self.input_regions, self.active_indices,
                            self.cage_experiments, self.feat_index)
        elif self.feature == 4:
            self.real_values = readRnaSeqFeature(self.directory, self.input_regions, self.active_indices,
                                                 self.feat_index)

    def displayFeatureProgress(feature, string_update):
        if myThread.print_init:
            sys.stdout.write(4*"\033[F")
        else:
            myThread.print_init = 1
        myThread.feature_str[feature] = string_update
        display_str = str()
        for i in range(1, 5):
            display_str += myThread.feature_str[i] + "\n"
        sys.stdout.write(display_str)
        sys.stdout.flush()

def readDnaseChipFeature(dnase_chipseq_dir, input_regions, active_indices):
    # List of files containing region indices with overlapping peak in different DNase-seq and ChIP-seq data
    dnase_chipseq_files = sorted(os.listdir(dnase_chipseq_dir))

    # Iterate through each file (each experiment in a specific cell/tissue-type and with a specific target if ChIP-seq)
    for i in range(len(dnase_chipseq_files)):
        dnase_chipseq_file = dnase_chipseq_files[i]
        try:
            regions = pd.read_table(dnase_chipseq_dir+dnase_chipseq_file, engine='c', header=None, squeeze=True).tolist()
            valid = list(input_regions.intersection(regions))
            for j in range(len(valid)):
                active_indices[valid[j]].append(i)
        except (pd.errors.EmptyDataError,pd.io.common.EmptyDataError) as _: # Some input file may be empty
            continue

        # Status output
        p = int((i+1)/len(dnase_chipseq_files)*100)
        display_str = "\tDNase-seq and ChIP-seq [" + "=" * p + " "*(100-p) + "] " + str(p)+"%\t"
        myThread.displayFeatureProgress(1, display_str)

def readChromHmmFeature(chromhmm_dir, input_regions, active_indices, chromhmm_num_states, feat_index):
    num_current_features = feat_index

    # List of files containing region indices and their overlapping ChromHMM state for each cell/tissue type
    chromhmm_files = sorted(os.listdir(chromhmm_dir))
    for i in range(len(chromhmm_files)): # Iterate through each file (each cell-type)
        chromhmm_file = chromhmm_files[i]
        if os.stat(chromhmm_dir+chromhmm_file).st_size == 0: # These files should not be empty
            print ('! Empty ChromHMM data file',chromhmm_file,i)
            continue
        with gzip.open(chromhmm_dir + chromhmm_file,'rb') as f:
            regions_found = 0
            for line in f:
                l = line.strip().split()
                region = int(l[0].decode('utf-8'))
                if region in input_regions:
                    regions_found += 1
                    state = l[1].decode('utf-8')
                    state = int(state[1:]) if state.startswith('U') else int(state)
                    active_indices[region].append(num_current_features+(state-1))

                # ChromHMM runtime optimization added 10 July 2019
                if regions_found == len(input_regions):
                    break
            num_current_features += chromhmm_num_states

        # Status output
        p = int((i+1)/len(chromhmm_files)*100)
        display_str = "\tChromHMM [" + "=" * p + " "*(100-p) + "] " + str(p)+"%"
        myThread.displayFeatureProgress(2, display_str)

def readCageFeature(cage_dir, input_regions, active_indices, cage_num_experiments, feat_index):
    num_current_features = feat_index

    # A file containing region indices and their CAGE peak data across multiple cell-types
    cage_file = os.listdir(cage_dir)[0]
    try:
        df = pd.read_table(cage_dir+cage_file,engine='c',header=None).as_matrix()
        regions = df[:,0] # Position indices
        features = df[:,1:] # Presence of CAGE peak in each region in each experiment
        for i in range(len(regions)):
            if regions[i] in input_regions:
                a = num_current_features + np.where(features[i,:]>0)[0] # active CAGE features
                for j in a:
                    active_indices[regions[i]].append(j)

            # Status output
            p = int((i+1)/len(regions)*100)
            display_str = "\tCAGE [" + "=" * p + " "*(100-p) + "] " + str(p)+"% {0}".format(len(features[0]))
            myThread.displayFeatureProgress(3, display_str)

        num_current_features += len(features[0])
    except (pd.errors.EmptyDataError,pd.io.common.EmptyDataError) as _:
        num_current_features += cage_num_experiments # CAGE data is empty but still need to keep track of feature index

def readRnaSeqFeature(rnaseq_dir, input_regions, active_indices, feat_index):
    num_current_features = feat_index
    real_values = defaultdict(list) # Key: region index, value: real value for the non-binary features

    # List of files containing region indices and their RNA-seq level in different cell-types
    rnaseq_files = sorted(os.listdir(rnaseq_dir))
    for i in range(len(rnaseq_files)): # Iterate through each file (each cell-type)
        rnaseq_file = rnaseq_files[i]
        last_pos = int()
        try:
            df = pd.read_table(rnaseq_dir+rnaseq_file,engine='c',header=None).as_matrix()
            regions = df[:,0] # Position indices
            signals = df[:,1] # RNA-seq level of each region in the current cell-type
            for j in range(len(regions)):
                if regions[j] in input_regions:
                    active_indices[regions[j]].append(num_current_features+i)
                    last_pos = num_current_features+i
                    real_values[regions[j]].append(signals[j])
            # Status output
            p = int((i+1)/len(rnaseq_files)*100)
            display_str = "\tRNA-seq [" + "=" * p + " "*(100-p) + "] " + str(p)+"% Pos last added {0}".format(last_pos)
            myThread.displayFeatureProgress(4, display_str)
        except (pd.errors.EmptyDataError,pd.io.common.EmptyDataError) as _:
            #print ('! Empty RNA-seq data file',rnaseq_file,i)
            continue

    return real_values

# Read and save features from separate gzipped files
def readFeatures(dnase_chipseq_dir,chromhmm_dir,cage_dir,rnaseq_dir,chromhmm_num_states,cage_num_experiments,input_regions):
    active_indices = defaultdict(list) # Key: region index, value: non-zero feature indices for the given region
    cage_file = os.listdir(cage_dir)[0]
    df = pd.read_table(cage_dir+cage_file,engine='c',header=None).as_matrix()
    features = df[:,1:] # presence of CAGE peak in each region in each cell-type

    # Feature indices are calcualted pre-emptively for multi-threading to append the correct active_indices values at
    # region keys
    chromhmm_index = len(os.listdir(dnase_chipseq_dir))
    cage_index = chromhmm_index + chromhmm_num_states*len(os.listdir(chromhmm_dir))
    rnaseq_index = cage_index + len(features[0]) # len(os.listdir(cage_dir))
    feature_indices = [0, 0, chromhmm_index, cage_index, rnaseq_index]

    ### Process DNase-seq and ChIP-seq features ###
    # This is the simplest type. We only care about whether there is an overlapping peak at the region
    thread1 = myThread(1, "Thread-1", 1, dnase_chipseq_dir, input_regions, active_indices,
                       chromhmm_num_states, cage_num_experiments, feature_indices)

    ### Process ChromHMM features ###
    # We do one-hot encoding for ChromHMM. For example, if there are 25 states and a region is overlapping
    # with state 5, we have a vector of length 25 with its 5th value set to 1 and the rest set to 0.
    thread2 = myThread(2, "Thread-2", 2, chromhmm_dir, input_regions, active_indices,
                       chromhmm_num_states, cage_num_experiments, feature_indices)

    ### Process CAGE features ###
    # Similarly to DNase-seq and ChIP-seq data, we only care about the presence of a peak for CAGE data.
    # However, the data from different cell-types come in one file so we do not iterate through multiple files here.
    thread3 = myThread(3, "Thread-3", 3, cage_dir, input_regions, active_indices,
                       chromhmm_num_states, cage_num_experiments, feature_indices)

    ### Process RNA-seq features ###
    thread4 = myThread(4, "Thread-4", 4, rnaseq_dir, input_regions, active_indices,
                       chromhmm_num_states, cage_num_experiments, feature_indices)

    # Begin the threads for processing features individually --calls run() for each thread instance
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()

    # Wait for all 4 threads to finish before returning the completed active_indices and real_values from RNA-seq
    # feature
    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()

    return active_indices,thread4.real_values

# Write aggregated features for each region
def writeFormattedFeatures(gzf,chrs,starts,ends,region_indices,active_indices,real_values):
    for i in range(len(region_indices)): # Iterate through each region to write its features
        ### Output format ###
        # Each line: chr start end pos_index | <list of all active/non-zero feature indices> | <list of real values>
        # Values in each list are separated by tab
        # Real values correspond to the last active/non-zero feature indices. For example, if there are n real
        # values, they correspond to the n last active/non-zero feature indices. If there is nothing written after
        # the second vertical bar (|), there is no feature with real values. All active/zero features are binary in this
        # case.

        # First, write chromosome, start, end, and region index for the sample
        tp = '\t'.join([str(s) for s in [chrs[i],starts[i],ends[i],region_indices[i]]])
        gzf.write(tp.encode())

        # Second, write active/non-zero feature indices
        t = active_indices[region_indices[i]]
        tp = '\t|'+'\t'.join([str(s) for s in t])
        gzf.write(tp.encode())

        # Third, write real/non-binary values (RNA-seq signal levels)
        t = real_values[region_indices[i]] # Real values
        tp = '\t|'+'\t'.join([str(round(s,10)) for s in t])+'\n'
        gzf.write(tp.encode())

        p = int((i+1)/len(region_indices)*100)
        sys.stdout.write("\r\t[" + "=" * p + " "*(100-p) + "] " + str(p)+"%")
        sys.stdout.flush()

    sys.stdout.write('\n')

# Added 11 July 2019 for multi-threading feature --each thread can add features at region keys out of sequence; sorting
# just for output
def sortActiveIndicesByPosition(active_indices):
    for key in active_indices.keys():
        active_indices[key].sort()

def main():
    description = 'Aggregate processed feature data into one file for a given set of genomic regions within one species'
    epilog = '# Example of generating data for the first 1 million human genomic regions: python \
    source/generateDataThreaded.py -p region/hg19.mm10.50bp.h.gz -ca feature/intersect/hg19_CAGE/ -ch \
    feature/intersect/hg19_ChromHMM/ -dn feature/intersect/hg19_DNaseChIPseq/ -rn feature/intersect/hg19_RNAseq/ -chn \
    25 -can 1829 -fn 8824 -o data/split/all_1.h.gz -s -c 1000000 -i 1'

    parser = argparse.ArgumentParser(prog='python source/generateDataThreaded.py', description=description, epilog=epilog)
    parser.add_argument('-s', '--split', action='store_true',
                        help='whether to split the data into multiple chunks to submit separate jobs (default: False)')
    parser.add_argument('-c', '--split-chunk-size', default=1000000, type=int,
                        help='size of each chunk if splitting (default: 1000000)')
    parser.add_argument('-i', '--split-index', type=int, default=1,
                        help='split index starting from 1 if splitting (default: 1)')

    g1 = parser.add_argument_group('required arguments')
    g1.add_argument('-p', '--region-filename', type=str, required=True,
                    help='path to species-specific output file (.h.gz or .m.gz) from samplePairs.py')
    g1.add_argument('-ca', '--cage-dir', type=str, required=True,
                    help='path to directory with output files from runIntersect for CAGE')
    g1.add_argument('-ch', '--chromhmm-dir', type=str, required=True,
                    help='path to directory with output files from runIntersect for ChromHMM')
    g1.add_argument('-dn', '--dnase-chipseq-dir', type=str, required=True,
                    help='path to directory with output files from runIntersect for DNase-seq and ChIP-seq')
    g1.add_argument('-rn', '--rnaseq-dir', type=str, required=True,
                    help='path to directory with output files from runIntersect for RNA-seq')
    g1.add_argument('-chn', '--chromhmm-num-states', type=int, required=True,
                    help='number of ChromHMM chromatin states (currently: 25 for human, 15 for mouse)')
    g1.add_argument('-can', '--cage-num-experiments', type=int, required=True,
                    help='number of CAGE experiments (currently: 1829 for human, 1073 for mouse)')
    g1.add_argument('-fn', '--num-features', type=int, required=True,
                    help='total number of features (currently: 8824 for human, 3313 for mouse)')
    g1.add_argument('-o', '--output-filename', type=str, required=True,
                    help='path to output file')
    args = parser.parse_args()

    ### Read regions ###
    print ('Reading regions...')
    if args.split:
        split_start = (args.split_index-1)*args.split_chunk_size
        split_end = (args.split_index)*args.split_chunk_size
        print ('\tPosition index ranges from %d to %d' % (split_start,split_end))

    df = pd.read_table(args.region_filename,engine='c',header=None,usecols=[0,1,2,3],names=['chr','start','end','index']).as_matrix()
    df = df[df[:,3].argsort()]
    if args.split:
        chrs = df[split_start:split_end,0] # chromsomes
        starts = df[split_start:split_end,1] # region start
        ends = df[split_start:split_end,2] # region end
        region_indices = df[split_start:split_end,3] # region indices
    else:
        chrs = df[:,0] # chromsomes
        starts = df[:,1] # region start
        ends = df[:,2] # region end
        region_indices = df[:,3] # region indices
    print('\t%s regions read' % len(region_indices))

    ### Read features of the regions of interest from multiple files and write to one output file ###
    with gzip.open(args.output_filename if args.output_filename.endswith('.gz') else args.output_filename+'.gz','wb') as gzf:

        # Read features
        print ('Reading features...')
        active_indices,real_values = readFeatures(args.dnase_chipseq_dir,
                                                  args.chromhmm_dir,
                                                  args.cage_dir,
                                                  args.rnaseq_dir,
                                                  args.chromhmm_num_states,
                                                  args.cage_num_experiments,
                                                  set(region_indices))
        regions_read = set(active_indices.keys())
        overlapping_regions = regions_read.intersection(region_indices)
        print ('\tOverlapping regions: %d' % len(overlapping_regions))

        # Threading causes features to finish out of order, so re-order for consistency
        sortActiveIndicesByPosition(active_indices)

        # Write formatted features
        print ('Writing formatted features...')
        writeFormattedFeatures(gzf,chrs,starts,ends,region_indices,active_indices,real_values)

main()