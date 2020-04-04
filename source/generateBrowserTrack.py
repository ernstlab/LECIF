'''
This script ensembles multiple predictions generated by multiple neural networks and formats the ensembled prediction
into a Bed file for viewing the score on a genome browser and conducting downstream analyses.
'''

import sys,gzip,numpy as np,argparse,pandas as pd

def main():
    epilog = '# Example: python source/generateBrowserTrack.py -p position/hg19.evenchr.mm10.50bp.gz -f NN/odd_ensemble/even_all_*.gz -o NN/odd_ensemble/even_all.gz'

    parser = argparse.ArgumentParser(prog='python source/generateBrowserTrack.py',
                                     description='Generate a genome browser track given multiple predictions generated by predict.py',
                                     epilog=epilog)
    g1 = parser.add_argument_group('required arguments specifying input and output')

    g1.add_argument('-p', '--region-filename', type=str, required=True,
                    help='path to the output file (.gz) from samplePairs.py')
    g1.add_argument('-f', '--score-filenames', help='list of output files from predict.py', type=str, nargs='+',
                    required=True)
    g1.add_argument('-o', '--output-filename', help='path to final output file', type=str, required=True)

    parser.add_argument('-a', '--index-to-use-for-chromosome',
                        help='column index to use for human chromosome in the specified region file', type=int,
                        default=0)
    parser.add_argument('-b', '--index-to-use-for-start',
                        help='column index to use for human region start position in the specified region file',
                        type=int, default=1)
    parser.add_argument('-c', '--index-to-use-for-end',
                        help='column index to use for human region end position in the specified region file', type=int,
                        default=11)
    args = parser.parse_args()
    print(args)

    # Read score positions
    a = pd.read_table(args.position_filename,engine='c',header=None,
                      usecols=[args.index_to_use_for_chromosome,args.index_to_use_for_start,args.index_to_use_for_end],
                      names=['chrom','start','end'])
    a_chrom = a['chrom'].tolist()
    a_start = a['start'].tolist()
    a_end = a['end'].tolist()

    print ('Reading scores...')
    f = pd.read_table(args.score_filenames[0],squeeze=True,header=None).values
    s = np.zeros((len(f),len(args.score_filenames)),dtype=float)
    s[:,0] = f
    for i in range(1,len(args.score_filenames)):
        s[:,i] = pd.read_table(args.score_filenames[i],squeeze=True,header=None).values
        perc = int((i+1)/len(args.score_filenames) * 100)
        sys.stdout.write("\r\tProgress [" + "="*perc + " "*(100-perc) + "]" + str(perc) + "%")
        sys.stdout.flush()

    print ('\nEnsembling scores...')
    a_score = np.mean(s,axis=1)

    print ('Writing the output file...')
    with gzip.open(args.output_filename,'wb') as f: # open output file
        for i in range(len(a_chrom)): # iterate through each position with scores
            l = '%s\t%d\t%d\t%.5f\n' % (a_chrom[i],a_start[i],a_end[i],a_score[i])
            f.write(l.encode())
            perc = int((i+1)/len(a_chrom) * 100)
            sys.stdout.write("\r\tProgress [" + "=" * perc + " " * (100 - perc) + "]" + str(perc) + "%")
            sys.stdout.flush()

main()