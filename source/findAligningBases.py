'''
This script finds mouse genomic bases that align (at the sequence level) to human genomic on a given human chromosome
based on axtNet alignment data. Note the following:
- axtNet alignment data coordinates start from 1, and here we convert it to the 0-based coordinate system used in bed
files.
- We process alignments between a positive strand in human and a negative strand in mouse differently than alignments
between positive strands because the coordinates provided for negative strands in mouse start from the end of the
chromosome. This requires knowing the size of all mouse chromosomes. More information is found here:
https://genome.ucsc.edu/goldenPath/help/axt.html
'''

import sys,gzip,argparse,os
from collections import defaultdict

def readChromSizes(fn):
    d = defaultdict(int)
    with open(fn) as f:
        for line in f:
            l = line.strip().split()
            chrom,length = l[0],int(l[1])
            d[chrom] = length
    return d

def main():
    description = 'For a given human chromosome, find all mouse bases that align to that human chromosome'
    epilog = '# Example for human chromosome 21: python source/findAligningBases.py \
    -a position/axtNet/chr21.hg19.mm10.net.axt.gz \
    -m position/mm10.chrom.sizes \
    -o position/aligning_bases_by_chrom/hg19.chr21.mm10.basepair.gz'
    parser = argparse.ArgumentParser(prog='python source/findAligningBases.py', description=description, epilog=epilog)
    g1 = parser.add_argument_group('required arguments')
    g1.add_argument('-a','--axtnet-filename', help='path to human-chromosome-specific axtNet filename', required=True, type=str)
    g1.add_argument('-m','--mouse-chrom-size-filename', help='path to mm10.chrom.sizes', required=True, type=str)
    g1.add_argument('-o','--output-filename', help='path to human-chromosome-specific output filename', required=True, type=str)
    args = parser.parse_args()

    # When using job arrays, human chromosome X equals human chromosome 23
    if '23' in args.axtnet_filename:
        args.axtnet_filename = args.axtnet_filename.replace('23','X')
        args.output_filename = args.output_filename.replace('23','X')
    args.output_filename += '.gz' if not args.output_filename.endswith('.gz') else '' # make sure output is .gz
    mm_chromosome_size = readChromSizes(args.mouse_chrom_size_filename)

    with gzip.open(args.axtnet_filename) as fin, gzip.open(args.output_filename,'wb') as fout:

        # Skip all comment lines at the top of the file and break when the comments end
        for line in fin:
            if line.startswith(b'#'):
                continue
            else:
                break

        while line:
            l = line.strip().split() # read first line
            if len(l)==0: # break if line is empty
                break

            # Read genomic coordinates of this particular chain
            align_num  = l[0].decode('utf-8')
            hg_chrom,hg_start,hg_end = l[1].decode('utf-8'),int(l[2]),int(l[3])
            mm_chrom,mm_start,mm_end,mm_strand = l[4].decode('utf-8'),int(l[5]),int(l[6]),l[7].decode("utf-8")=='+'

            # Exclude alignments with mouse chrM or chrY
            if len(mm_chrom)>5 or 'M' in mm_chrom or 'Y' in mm_chrom:
                for i in range(4): # in order to skip this particular chain, need to skip 4 lines
                    line = fin.readline()
                continue

            # Shift everything in human coordinates by 1 bp because dumb differences in coordinate systems
            hg_start -= 1
            hg_end -= 1

            if mm_strand:
                mm_start -= 1
                mm_end -= 1

            # If the aligning mouse region is on the negative strand, we need to recompute coordinates
            else:
                mm_chrom_size = mm_chromosome_size[mm_chrom]
                s = mm_chrom_size-mm_end
                e = mm_chrom_size-mm_start
                mm_start = s
                mm_end = e

            # Read actual sequence of A,C,T, and G in the following lines
            hg_seq = fin.readline().strip().decode('utf-8')
            mm_seq = fin.readline().strip().decode('utf-8')

            for i in range(len(hg_seq)):
                hg_gap = hg_seq[i]=='-' # set to True if there's a gap in human
                mm_gap = mm_seq[i]=='-' # set to True if there's a gap in mouse

                # 1. Gaps in both species
                if hg_gap and mm_gap:
                    hg_start += 1
                    if mm_strand:
                        mm_start += 1
                    else:
                        mm_end -= 1

                # 2. Gap in mouse only
                elif not hg_gap and mm_gap:
                    hg_start += 1


                # 3. Gap in human only
                elif hg_gap and not mm_gap:
                    if mm_strand:
                        mm_start += 1
                    else:
                        mm_end -= 1

                # 4. No gap
                else:
                    if mm_strand:
                        toprint = '\t'.join([str(s) for s in [hg_chrom,hg_start,hg_start+1,hg_seq[i],mm_chrom,mm_start,mm_start+1,mm_seq[i]]])+'\n'
                        fout.write(toprint.encode())
                        hg_start += 1
                        mm_start += 1
                    else:
                        toprint = '\t'.join([str(s) for s in [hg_chrom,hg_start,hg_start+1,hg_seq[i],mm_chrom,mm_end,mm_end+1,mm_seq[i]]])+'\n'
                        fout.write(toprint.encode())
                        hg_start += 1
                        mm_end -= 1 # move backwards

                #print ('\t',i,hg_start,mm_start,mm_end,hg_gap,mm_gap,hg_seq[i],mm_seq[i])

            fin.readline() # read a black line before the next alignment
            line = fin.readline() # read and save next line

            #sys.stdout.write("\rCurrent chain: "+align_num)
            #sys.stdout.flush()

    # Sort
    tmp_filename = args.output_filename+'.tmp'
    os.system("gzip -cd %s |  sort -k1,1 -k2,2n | gzip > %s " % (args.output_filename,tmp_filename))
    os.system("mv %s %s" % (tmp_filename,args.output_filename))

main()
