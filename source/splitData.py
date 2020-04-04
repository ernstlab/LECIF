'''
This script splits input data for training, validation, test, and model comparison based on human and mouse chromosomes
to ensure no data leakage, given which sets of chromosomes should be used for which purpose specified in an input file
or as input arguments
'''
import gzip,argparse

class LoadFromFile (argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            contents = f.read()

        data = parser.parse_args(contents.split(), namespace=namespace)
        for k, v in vars(data).items():
            if v and k != option_string.lstrip('-'):
                setattr(namespace, k, v)

def main():
    description = 'Split input data for training, validation, test, and model comparison based on human and mouse chromosomes'
    epilog = '# Example: python source/splitData.py \
    -A data/all.h.gz -B data/all.m.gz -N 32285361 -o data/ @example/splitData_args.txt'

    parser = argparse.ArgumentParser(prog='python source/splitData.py', description=description, epilog=epilog)
    
    g1 = parser.add_argument_group('required arguments specifying input and output')
    
    g1.add_argument('-A','--human-feature-data-filename',type=str,required=True,
                    help='path to combined human feature data filename')
    g1.add_argument('-B','--mouse-feature-data-filename',type=str,required=True,
                    help='path to combined mouse feature data filename')
    g1.add_argument('-N','--total-num-examples',type=int,required=True,
                    help='total number of examples/pairs')
    g1.add_argument('-o','--output-dir',type=str,required=True)
    
    g2 = parser.add_argument_group('required arguments specifying the split (may be specified in an input file --see example below')

    g2.add_argument('--odd-training-human-chrom',required=True,
                    help='human chromosomes to include in odd training data',type=str,nargs='+')
    g2.add_argument('--odd-validation-human-chrom',required=True,
                    help='human chromosomes to include in odd validation data',type=str,nargs='+')
    g2.add_argument('--odd-test-human-chrom',required=True,
                    help='human chromosomes to include in odd test data', type=str, nargs='+')
    g2.add_argument('--odd-prediction-human-chrom',required=True,
                    help='human chromosomes to include in odd prediction data',type=str,nargs='+')

    g2.add_argument('--odd-training-mouse-chrom',required=True,
                    help='mouse chromosomes to include in odd training data',type=str,nargs='+')
    g2.add_argument('--odd-validation-mouse-chrom',required=True,
                    help='mouse chromosomes to include in odd validation data',type=str,nargs='+')
    g2.add_argument('--odd-test-mouse-chrom',required=True,
                    help='mouse chromosomes to include in odd test data', type=str, nargs='+')

    g2.add_argument('--even-training-human-chrom',required=True,
                    help='human chromosomes to include in even training data',type=str,nargs='+')
    g2.add_argument('--even-validation-human-chrom',required=True,
                    help='human chromosomes to include in even validation data',type=str,nargs='+')
    g2.add_argument('--even-test-human-chrom',required=True,
                    help='human chromosomes to include in even test data',type=str,nargs='+')
    g2.add_argument('--even-prediction-human-chrom',required=True,
                    help='human chromosomes to include in even prediction data',type=str,nargs='+')

    g2.add_argument('--even-training-mouse-chrom',required=True,
                    help='mouse chromosomes to include in even training data',type=str,nargs='+')
    g2.add_argument('--even-validation-mouse-chrom',required=True,
                    help='mouse chromosomes to include in even validation data',type=str,nargs='+')
    g2.add_argument('--even-test-mouse-chrom',required=True,
                    help='mouse chromosomes to include in even test data',type=str,nargs='+')

    g2.add_argument('--held-out-human-chrom',required=True,
                    help='human chromosomes held out from training, validation, and test for model comparison',
                    type=str,nargs='+')
    g2.add_argument('--held-out-mouse-chrom',required=True,
                    help='mouse chromosomes held out from training, validation, and test for model comparison',
                    type=str,nargs='+')

    args = parser.parse_args()

    # Convert lists of chromosomes into sets
    odd_training_human_chrom = set(args.odd_training_human_chrom)
    odd_validation_human_chrom = set(args.odd_validation_human_chrom)
    odd_test_human_chrom = set(args.odd_test_human_chrom)
    odd_prediction_human_chrom = set(args.odd_prediction_human_chrom)

    odd_training_mouse_chrom = set(args.odd_training_mouse_chrom)
    odd_validation_mouse_chrom = set(args.odd_validation_mouse_chrom)
    odd_test_mouse_chrom = set(args.odd_test_mouse_chrom)

    even_training_human_chrom = set(args.even_training_human_chrom)
    even_validation_human_chrom = set(args.even_validation_human_chrom)
    even_test_human_chrom = set(args.even_test_human_chrom)
    even_prediction_human_chrom = set(args.even_prediction_human_chrom)

    even_training_mouse_chrom = set(args.even_training_mouse_chrom)
    even_validation_mouse_chrom = set(args.even_validation_mouse_chrom)
    even_test_mouse_chrom = set(args.even_test_mouse_chrom)

    held_out_human_chrom = set(args.held_out_human_chrom)
    held_out_mouse_chrom = set(args.held_out_mouse_chrom)

    ### Start reading and writing
    hin = gzip.open(args.human_feature_filename)
    min = gzip.open(args.mouse_feature_filename)

    with gzip.open(args.output_dir+'/odd_training.h.gz','wb') as human_odd_training_out,\
        gzip.open(args.output_dir+'/odd_validation.h.gz','wb') as human_odd_validation_out, \
        gzip.open(args.output_dir+'/odd_test.h.gz','wb') as human_odd_test_out, \
        gzip.open(args.output_dir+'/odd_all.h.gz','wb') as human_odd_prediction_out,\
        gzip.open(args.output_dir+'/even_training.h.gz','wb') as human_even_training_out,\
        gzip.open(args.output_dir+'/even_validation.h.gz','wb') as human_even_validation_out, \
        gzip.open(args.output_dir+'/even_test.h.gz','wb') as human_even_test_out, \
        gzip.open(args.output_dir+'/even_all.h.gz', 'wb') as human_even_prediction_out, \
        gzip.open(args.output_dir+'/held_out.h.gz','wb') as human_held_out_out, \
        gzip.open(args.output_dir+'/odd_training.m.gz','wb') as mouse_odd_training_out,\
        gzip.open(args.output_dir+'/odd_validation.m.gz','wb') as mouse_odd_validation_out, \
        gzip.open(args.output_dir+'/odd_test.m.gz','wb') as mouse_odd_test_out, \
        gzip.open(args.output_dir+'/odd_all.m.gz','wb') as mouse_odd_prediction_out,\
        gzip.open(args.output_dir+'/even_training.m.gz','wb') as mouse_even_training_out,\
        gzip.open(args.output_dir+'/even_validation.m.gz','wb') as mouse_even_validation_out, \
        gzip.open(args.output_dir+'/even_test.m.gz','wb') as mouse_even_test_out, \
        gzip.open(args.output_dir+'/even_all.m.gz', 'wb') as mouse_even_prediction_out, \
        gzip.open(args.output_dir+'/held_out.m.gz','wb') as mouse_held_out_out:

        for i in range(args.total_num_examples):
            hline = hin.readline()
            mline = min.readline()
            hl = hline.strip().split()
            ml = mline.strip().split()
            hg_chrom = hl[0].decode('utf-8')
            mm_chrom = ml[0].decode('utf-8')

            # Write lines if the current human chromosome is one of the specified chromosomes
            if hg_chrom in odd_prediction_human_chrom:
                human_odd_prediction_out.write(hline)
                mouse_odd_prediction_out.write(mline)

                if hg_chrom in odd_training_human_chrom and mm_chrom in odd_training_mouse_chrom:
                    human_odd_training_out.write(hline)
                    mouse_odd_training_out.write(mline)

                if hg_chrom in odd_validation_human_chrom and mm_chrom in odd_validation_mouse_chrom:
                    human_odd_validation_out.write(hline)
                    mouse_odd_validation_out.write(mline)

                if hg_chrom in odd_test_human_chrom and mm_chrom in odd_test_mouse_chrom:
                    human_odd_test_out.write(hline)
                    mouse_odd_test_out.write(mline)

            if hg_chrom in even_prediction_human_chrom:
                human_even_prediction_out.write(hline)
                mouse_even_prediction_out.write(mline)

                if hg_chrom in even_training_human_chrom and mm_chrom in even_training_mouse_chrom:
                    human_even_training_out.write(hline)
                    mouse_even_training_out.write(mline)

                if hg_chrom in even_validation_human_chrom and mm_chrom in even_validation_mouse_chrom:
                    human_even_validation_out.write(hline)
                    mouse_even_validation_out.write(mline)

                if hg_chrom in even_test_human_chrom and mm_chrom in even_test_mouse_chrom:
                    human_even_test_out.write(hline)
                    mouse_even_test_out.write(mline)

            if hg_chrom in held_out_human_chrom and mm_chrom in held_out_mouse_chrom:
                human_held_out_out.write(hline)
                mouse_held_out_out.write(mline)

main()