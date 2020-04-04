'''
This script samples the first base of every non-overlapping genomic window defined across consecutive human genomic
bases that align to mouse, converting aligning pairs of human and mouse bases into aligning pairs of human and mouse
regions.
The main output file (ending with .gz) has 18 columns:
- Columns 1-4 (chrom, start, end, nucleotide) correspond to the first human base of the 50-bp window
- Columns 5-8 correspond to the aligning mouse base of the start of the 50-bp block
- Column 9 is the pair index of the start of the 50-bp block
- Columns 10-13 correspond to the human of the end of the 50-bp block
- Columns 14-17 correspond to the aligning mouse base of the end of the 50-bp block
- Coummn 18 is the pair index of the end of the 50-bp block
We also generate separate output files for human regions (file ending with .h.gz) and mouse regions (file ending  with
.m.gz)
'''

import gzip,argparse,os

def main():
    description = 'Sample the first base of every non-overlapping genomic window defined across consecutive human \
    genomic bases that align to mouse'
    epilog = '# Example: python source/samplePairs.py -i position/hg19.mm10.basepair.gz -o position/hg19.mm10.50bp'

    parser = argparse.ArgumentParser(prog='python source/samplePairs.py',
                                     description=description,
                                     epilog=epilog)
    g1 = parser.add_argument_group('required arguments')
    g1.add_argument('-i','--input-filename',
                    help='path to output filename from aggregateAligningBases containing aligning pairs of human and mouse bases',
                    required=True,type=str)
    parser.add_argument('-b','--bin-size',
                    help='size (bp) of the non-overlapping genomic window (default: 50)',
                    type=int,default=50)
    g1.add_argument('-o','--output-prefix',
                    help='prefix for output files',
                    required=True,type=str)
    args = parser.parse_args()

    with gzip.open(args.input_filename) as fin, gzip.open(args.output_prefix+'.gz','wb') as fout:

        # Write the very first line
        line = fin.readline().strip()
        fout.write(line+b'\t')
        l = line.split()
        hg_chrom = l[0].decode('utf-8')
        hg_pos = int(l[1])

        i = 1
        prev_chrom = hg_chrom
        prev_pos = hg_pos
        prev_line = line

        for line in fin: # Iterate
            line = line.strip()
            l = line.split()
            hg_chrom = l[0].decode('utf-8')
            hg_pos = int(l[1])

            # Current base is contiguous to the previous base
            if prev_chrom == hg_chrom and hg_pos == prev_pos + 1:
                if i%args.bin_size==0:
                    fout.write(prev_line+b'\n')
                    fout.write(line+b'\t')
                i += 1

            # Current base is a start of new block
            else:
                fout.write(prev_line+b'\n')
                fout.write(line+b'\t')
                i = 1

            # Store current data
            prev_chrom = hg_chrom
            prev_pos = hg_pos
            prev_line = line

        fout.write(line)

    # Write separate files for human and mouse regions
    os.system("gzip -cd %s.gz | awk -v OFS='\t' '{print $1,$2,$3,$9}' | gzip > %s.h.gz" % (args.output_prefix,args.output_prefix))
    os.system("gzip -cd %s.gz | awk -v OFS='\t' '{print $5,$6,$7,$9}' | sort -k1,1 -k2,2n | gzip > %s.m.gz" % (args.output_prefix,args.output_prefix))

main()
