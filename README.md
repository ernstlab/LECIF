# LECIF: Learning Evidence of Conservation from Integrated Functional genomic annotations
LECIF is a supervised machine learning method that learns a genome-wide score of evidence for conservation at the functional genomics level. To learn the score, LECIF trains an ensemble of neural networks using a large set of functional genomic annotations from a pair of species with labels from their sequence alignments. 

## Human-Mouse LECIF score

### LECIF v1.1
LECIF score for human (hg19) and mouse (mm10) is available in BigWig format (.bw) [here](https://public.hoffman2.idre.ucla.edu/ernst/R0RG6/LECIF/hg19.LECIFv1.1.bw). The score is defined based on hg19 genomic coordinates. When displaying the score on [UCSC Genome Browser](https://genome.ucsc.edu/cgi-bin/hgGateway), to view the genomic bases in which the score is available, display [Net Track for mouse](https://genome.ucsc.edu/cgi-bin/hgTables?db=hg19&hgta_group=compGeno&hgta_track=placentalChainNet&hgta_table=netMm10&hgta_doSchema=describe+table+schema), which is a subtrack in Placental Chain/Net Track under Comparative Genomics.

LECIF score mapped to hg38, mm10, and mm39 genomic coordinates are also available in BigWig format below. To map the score from hg19 to mm10, we relied on the output file `position/hg19.mm10.basepair.gz` from Step 1 below that lists aligning bases between hg19 and mm10. To map the score from hg19 to hg38 or from mm10 to mm39, we used UCSC Genome Browser's [liftOver](https://genome.ucsc.edu/cgi-bin/hgLiftOver) tool. We note that when we encountered multiple hg19 bases mapping to the same base in a new genome/assembly, we assigned the average LECIF score of the multiple hg19 bases to the base in the new genome/assembly.
- [hg38](https://public.hoffman2.idre.ucla.edu/ernst/R0RG6/LECIF/hg38.LECIFv1.1.bw)
- [mm10](https://public.hoffman2.idre.ucla.edu/ernst/R0RG6/LECIF/mm10.LECIFv1.1.bw)
- [mm39](https://public.hoffman2.idre.ucla.edu/ernst/R0RG6/LECIF/mm39.LECIFv1.1.bw)

### v1.1 updates
- A bug fix correcting how RNA-seq input features are processed in `source/predict.py`
- Missing human RNA-seq datasets added back 

The first version of the score (LECIF v1) before the updates are available [here](https://public.hoffman2.idre.ucla.edu/ernst/R0RG6/LECIF/). The two versions are highly correlated with a Pearson correlation coefficient of 0.97.

## Applying LECIF to human and mouse
### Requirements
LECIF was run in a Linux system (CentOS release 6.10). No installation is needed as long as you have all the scripts in the [source](source/) directory and the following resources:
1. [Python 3](https://www.python.org/downloads/)
2. [Scipy](https://www.scipy.org/), [Numpy](http://www.numpy.org/)
3. [PyTorch 0.3.0.post4](https://pytorch.org/get-started/previous-versions/)
4. [scikit-learn 0.19.1](https://scikit-learn.org/stable/) 
5. [Bedtools](https://bedtools.readthedocs.io/en/latest/content/bedtools-suite.html) 
6. [bigWigToBedGraph](http://hgdownload.soe.ucsc.edu/admin/exe/), [bedGraphToBigWig](http://hgdownload.soe.ucsc.edu/admin/exe/)

Although not required, given the large number of genomic regions and functional genomic data sets used here, job arrays are ___highly___ recommended to parallelize almost every step. 

### Example files
- An example input file for Step 2.6 is provided [here](example/splitData_args.txt).
- Examples of processed files generated at the end of Step 3 are provided as gzipped files [here](example/). These were downsampled from the actual processed files and can be directly used as input files to [train.py](source/train.py) to train a neural network in Step 4, which should take ~5 minutes on average.

All data used to learn the LECIF score is publically available as described in the following steps.

### Step 1. Find aligning genomic regions

1. Download [axtNet files](http://hgdownload.cse.ucsc.edu/goldenpath/hg19/vsMm10/axtNet/) with human as the reference. These files describe chained and netted alignments between human and mouse.

2. For each human chromosome (except for Y and mitochondrial chromosomes), find all mouse bases that align to that human chromosome. In addition to the axtNet files, this requires file [mm10.chrom.sizes](http://hgdownload.cse.ucsc.edu/goldenpath/mm10/bigZips/mm10.chrom.sizes) as input.

		usage: python source/findAligningBases.py [-h] -a AXTNET_FILENAME -m
							  MOUSE_CHROM_SIZE_FILENAME -o
							  OUTPUT_FILENAME

		For a given human chromosome, find all mouse bases that align to that human
		chromosome
		
		optional arguments:
		  -h, --help            show this help message and exit
		
		required arguments:
		  -a AXTNET_FILENAME, --axtnet-filename AXTNET_FILENAME
		                        path to human-chromosome-specific axtNet filename
		  -m MOUSE_CHROM_SIZE_FILENAME, --mouse-chrom-size-filename MOUSE_CHROM_SIZE_FILENAME
		                        path to mm10.chrom.sizes
		  -o OUTPUT_FILENAME, --output-filename OUTPUT_FILENAME
		                        path to human-chromosome-specific output filename
		
		# Example for human chromosome 21: python source/findAligningBases.py -a
		position/axtNet/chr21.hg19.mm10.net.axt.gz -m position/mm10.chrom.sizes -o
		position/aligning_bases_by_chrom/hg19.chr21.mm10.basepair.gz


3. Combine all sorted aligning pairs into one file and assign a unique index to each.
 
		source/aggregateAligningBases \
		  <path to directory with output files from sortAligningBases> \
		  <path to output filename> 
		
		# Example:
		source/aggregateAligningBases position/aligning_bases_by_chrom/ position/hg19.mm10.basepair.gz
		
	The directory containing all the output files from `sortAligningBases` should contain nothing else.

4. Sample the first base of every non-overlapping genomic window of length 50 bp (at most) defined across consecutive bases in each human chromosome that align to mouse.

		usage: python source/samplePairs.py [-h] -i INPUT_FILENAME [-b BIN_SIZE] -o
						    OUTPUT_PREFIX

		Sample the first base of every non-overlapping genomic window defined across
		consecutive human genomic bases that align to mouse
		
		optional arguments:
		  -h, --help            show this help message and exit
		  -b BIN_SIZE, --bin-size BIN_SIZE
		                        size (bp) of the non-overlapping genomic window
		                        (default: 50)
		
		required arguments:
		  -i INPUT_FILENAME, --input-filename INPUT_FILENAME
		                        path to output filename from aggregateAligningBases
		                        containing aligning pairs of human and mouse bases
		  -o OUTPUT_PREFIX, --output-prefix OUTPUT_PREFIX
		                        prefix for output files
		
		# Example: python source/samplePairs.py -i position/hg19.mm10.basepair.gz -o
		position/hg19.mm10.50bp

										
	As output, three genomic region files are generated: (i) a file with human and mouse regions that align to each other, (ii) a file with human regions only, and (iii) a file with mouse regions only (e.g. `hg19.mm10.50bp.gz`, `hg19.mm10.50bp.h.gz`, and `hg19.mm10.50bp.m.gz`, respectively). 

5. Split the pairs based on whether the human region lies on an odd or even chromosome (X chromosome counts as even) for later use.

		gzip -cd position/hg19.mm10.50bp.gz |\
		awk -v OFS="\t"  '{sub("chr", "" ,$1); print $0}' |\
		awk -v OFS="\t" '$1 % 2 {print "chr"$0}' | gzip > position/hg19.oddchr.mm10.50bp.gz
		
		gzip -cd position/hg19.mm10.50bp.gz |\
		awk -v OFS="\t"  '{sub("chr", "" ,$1); print $0}' |\
		awk -v OFS="\t" '($1 % 2 == 0) {print "chr"$0}' | gzip > position/hg19.evenchr.mm10.50bp.gz

### Step 2. Generate input data

1. Download all the functional genomic annotations to include as features. Commands like `xargs -n 1 curl -O -L < files.txt` may be useful when downloading multiple files at once. [This table](table/SupplementaryTable1.xlsx) lists the annotations that we downloaded. 

	For each species, we store the annotation files in four separate directories to group files that need the same preprocessing steps:
	
	
	| Directory for human | Directory for mouse | Directory contains |
	| ------------------- | ------------------- | ------------------ |
	| hg19_DNaseChIPseq | mm10_DNaseqChIPseq | Bed files containing peak calls from DNase-seq and ChIP-seq experiments |
	| hg19_ChromHMM | mm10_ChromHMM | Bed files containing segmentation from ChromHMM chromatin state annotation |
	| hg19_CAGE | mm10_CAGE | a file containing a matrix of peak calls from CAGE experiments, with each row corresponding to a genomic location and each column corresponding to one CAGE experiment |
	| hg19_RNAseq | mm10_RNAseq | BigWig files containing signals from RNA-seq experiments |

2. For each of the eight directories, preprocess all the downloaded files in the directory.

		source/preprocessFeatureFiles
		  <path to input directory> \
		  <path to output directory> \
		  <species (0=Human, 1=Mouse)> \
		  <data type (0=DNase/ChIP-seq, 1=ChromHMM, 2=CAGE, 3=RNA-seq)>							
												
		# Example for human DNase-seq experiments:
		source/preprocessFeatureFiles \
		  feature/raw/hg19_DNaseChIPseq \
		  feature/preprocessed/hg19_DNaseChIPseq \
		  0 \
		  0 
										
3. For each preprocessed feature file, identify which genomic regions overlap the peaks or signals in that file.

		source/runIntersect	
		  <human/mouse coordinate filename from samplePairs.py> \
		  <output filename from preprocessFeatureFiles> \
		  <output filename to store output from BedTools intersect> \
		  <data type (0=DNase/ChIP-seq, 1=ChromHMM/CAGE/RNA-seq)>
							
		# Example for a human feature:
		source/runIntersect \
		  position/hg19.mm10.50bp.h.gz \
		  feature/preprocessed/E001-H3K27me3.narrowPeak.gz \
		  feature/intersect/E001-H3K27me3.narrowPeak.gz \
		  0
							
4. For each species, aggregate the preprocessed feature data for 1 million genomic regions (i.e. one chunk) at a time. This script involves multithreading of 4 threads.

		usage: python source/generateDataThreaded.py [-h] [-s] [-c SPLIT_CHUNK_SIZE]
		                                             [-i SPLIT_INDEX] -p
		                                             REGION_FILENAME -ca CAGE_DIR -ch
		                                             CHROMHMM_DIR -dn
		                                             DNASE_CHIPSEQ_DIR -rn RNASEQ_DIR
		                                             -chn CHROMHMM_NUM_STATES -can
		                                             CAGE_NUM_EXPERIMENTS -fn
		                                             NUM_FEATURES -o OUTPUT_FILENAME
		
		Aggregate processed feature data into one file for a given set of genomic
		regions within one species
		
		optional arguments:
		  -h, --help            show this help message and exit
		  -s, --split           whether to split the data into multiple chunks to
		                        submit separate jobs (default: False)
		  -c SPLIT_CHUNK_SIZE, --split-chunk-size SPLIT_CHUNK_SIZE
		                        size of each chunk if splitting (default: 1000000)
		  -i SPLIT_INDEX, --split-index SPLIT_INDEX
		                        split index starting from 1 if splitting (default: 1)
		
		required arguments:
		  -p REGION_FILENAME, --region-filename REGION_FILENAME
		                        path to species-specific output file (.h.gz or .m.gz)
		                        from samplePairs.py
		  -ca CAGE_DIR, --cage-dir CAGE_DIR
		                        path to directory with output files from runIntersect
		                        for CAGE
		  -ch CHROMHMM_DIR, --chromhmm-dir CHROMHMM_DIR
		                        path to directory with output files from runIntersect
		                        for ChromHMM
		  -dn DNASE_CHIPSEQ_DIR, --dnase-chipseq-dir DNASE_CHIPSEQ_DIR
		                        path to directory with output files from runIntersect
		                        for DNase-seq and ChIP-seq
		  -rn RNASEQ_DIR, --rnaseq-dir RNASEQ_DIR
		                        path to directory with output files from runIntersect
		                        for RNA-seq
		  -chn CHROMHMM_NUM_STATES, --chromhmm-num-states CHROMHMM_NUM_STATES
		                        number of ChromHMM chromatin states (currently: 25 for
		                        human, 15 for mouse)
		  -can CAGE_NUM_EXPERIMENTS, --cage-num-experiments CAGE_NUM_EXPERIMENTS
		                        number of CAGE experiments (currently: 1829 for human,
		                        1073 for mouse)
		  -fn NUM_FEATURES, --num-features NUM_FEATURES
		                        total number of features (currently: 8824 for human,
		                        3313 for mouse)
		  -o OUTPUT_FILENAME, --output-filename OUTPUT_FILENAME
		                        path to output file
		
		# Example of generating data for the first 1 million human genomic regions:
		python source/generateDataThreaded.py -p region/hg19.mm10.50bp.h.gz -ca
		feature/intersect/hg19_CAGE/ -ch feature/intersect/hg19_ChromHMM/ -dn
		feature/intersect/hg19_DNaseChIPseq/ -rn feature/intersect/hg19_RNAseq/ -chn
		25 -can 1829 -fn 8824 -o data/split/all_1.h.gz -s -c 1000000 -i 1

	Using the `-s` option allows you to process one chunk of data at a time to parallelize this step. Split index specified by `-i` determines which chunk should be processed. For example, if there are 5,500,000 regions and you wish to process 1,000,000 regions in each job, 6 separate jobs should be submitted with the same set of input arguments except for the split index which would range from 1 to 51. The last chunk would correspond to 500,000 regions.
	
	Note: If your species of interest lacks CAGE data, try the following:
	
	- Create a directory with an empty text file: `mkdir CAGE; touch CAGE/dummy.txt`
	- Provide that as CAGE directory and set the number of CAGE experiments to 0: `--cage-dir CAGE --cage-num-experiments 0`
									
5. For each species, combine the output from `generateDataThreaded.py` from each chunk into a single file.

		cat <path to output filenames from generatDataThreaded.py for one species> > <path to output filename>
		
		# Example for human:
		cat data/split/all_*.h.gz > data/all.h.gz

6. Split input data for training, validation, test, and model comparison based on human and mouse chromosomes to ensure no data leakage.

		usage: python source/splitData.py [-h] -A HUMAN_FEATURE_DATA_FILENAME -B
		                                  MOUSE_FEATURE_DATA_FILENAME -N
		                                  TOTAL_NUM_EXAMPLES -o OUTPUT_DIR
		                                  --odd-training-human-chrom
		                                  ODD_TRAINING_HUMAN_CHROM
		                                  [ODD_TRAINING_HUMAN_CHROM ...]
		                                  --odd-validation-human-chrom
		                                  ODD_VALIDATION_HUMAN_CHROM
		                                  [ODD_VALIDATION_HUMAN_CHROM ...]
		                                  --odd-test-human-chrom ODD_TEST_HUMAN_CHROM
		                                  [ODD_TEST_HUMAN_CHROM ...]
		                                  --odd-prediction-human-chrom
		                                  ODD_PREDICTION_HUMAN_CHROM
		                                  [ODD_PREDICTION_HUMAN_CHROM ...]
		                                  --odd-training-mouse-chrom
		                                  ODD_TRAINING_MOUSE_CHROM
		                                  [ODD_TRAINING_MOUSE_CHROM ...]
		                                  --odd-validation-mouse-chrom
		                                  ODD_VALIDATION_MOUSE_CHROM
		                                  [ODD_VALIDATION_MOUSE_CHROM ...]
		                                  --odd-test-mouse-chrom ODD_TEST_MOUSE_CHROM
		                                  [ODD_TEST_MOUSE_CHROM ...]
		                                  --even-training-human-chrom
		                                  EVEN_TRAINING_HUMAN_CHROM
		                                  [EVEN_TRAINING_HUMAN_CHROM ...]
		                                  --even-validation-human-chrom
		                                  EVEN_VALIDATION_HUMAN_CHROM
		                                  [EVEN_VALIDATION_HUMAN_CHROM ...]
		                                  --even-test-human-chrom
		                                  EVEN_TEST_HUMAN_CHROM
		                                  [EVEN_TEST_HUMAN_CHROM ...]
		                                  --even-prediction-human-chrom
		                                  EVEN_PREDICTION_HUMAN_CHROM
		                                  [EVEN_PREDICTION_HUMAN_CHROM ...]
		                                  --even-training-mouse-chrom
		                                  EVEN_TRAINING_MOUSE_CHROM
		                                  [EVEN_TRAINING_MOUSE_CHROM ...]
		                                  --even-validation-mouse-chrom
		                                  EVEN_VALIDATION_MOUSE_CHROM
		                                  [EVEN_VALIDATION_MOUSE_CHROM ...]
		                                  --even-test-mouse-chrom
		                                  EVEN_TEST_MOUSE_CHROM
		                                  [EVEN_TEST_MOUSE_CHROM ...]
		                                  --held-out-human-chrom HELD_OUT_HUMAN_CHROM
		                                  [HELD_OUT_HUMAN_CHROM ...]
		                                  --held-out-mouse-chrom HELD_OUT_MOUSE_CHROM
		                                  [HELD_OUT_MOUSE_CHROM ...]
		
		Split input data for training, validation, test, and model comparison based on
		human and mouse chromosomes
		
		optional arguments:
		  -h, --help            show this help message and exit
		
		required arguments specifying input and output:
		  -A HUMAN_FEATURE_DATA_FILENAME, --human-feature-data-filename HUMAN_FEATURE_DATA_FILENAME
		                        path to combined human feature data filename
		  -B MOUSE_FEATURE_DATA_FILENAME, --mouse-feature-data-filename MOUSE_FEATURE_DATA_FILENAME
		                        path to combined mouse feature data filename
		  -N TOTAL_NUM_EXAMPLES, --total-num-examples TOTAL_NUM_EXAMPLES
		                        total number of examples/pairs
		  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
		
		required arguments specifying the split (may be specified in an input file --see example below:
		  --odd-training-human-chrom ODD_TRAINING_HUMAN_CHROM [ODD_TRAINING_HUMAN_CHROM ...]
		                        human chromosomes to include in odd training data
		  --odd-validation-human-chrom ODD_VALIDATION_HUMAN_CHROM [ODD_VALIDATION_HUMAN_CHROM ...]
		                        human chromosomes to include in odd validation data
		  --odd-test-human-chrom ODD_TEST_HUMAN_CHROM [ODD_TEST_HUMAN_CHROM ...]
		                        human chromosomes to include in odd test data
		  --odd-prediction-human-chrom ODD_PREDICTION_HUMAN_CHROM [ODD_PREDICTION_HUMAN_CHROM ...]
		                        human chromosomes to include in odd prediction data
		  --odd-training-mouse-chrom ODD_TRAINING_MOUSE_CHROM [ODD_TRAINING_MOUSE_CHROM ...]
		                        mouse chromosomes to include in odd training data
		  --odd-validation-mouse-chrom ODD_VALIDATION_MOUSE_CHROM [ODD_VALIDATION_MOUSE_CHROM ...]
		                        mouse chromosomes to include in odd validation data
		  --odd-test-mouse-chrom ODD_TEST_MOUSE_CHROM [ODD_TEST_MOUSE_CHROM ...]
		                        mouse chromosomes to include in odd test data
		  --even-training-human-chrom EVEN_TRAINING_HUMAN_CHROM [EVEN_TRAINING_HUMAN_CHROM ...]
		                        human chromosomes to include in even training data
		  --even-validation-human-chrom EVEN_VALIDATION_HUMAN_CHROM [EVEN_VALIDATION_HUMAN_CHROM ...]
		                        human chromosomes to include in even validation data
		  --even-test-human-chrom EVEN_TEST_HUMAN_CHROM [EVEN_TEST_HUMAN_CHROM ...]
		                        human chromosomes to include in even test data
		  --even-prediction-human-chrom EVEN_PREDICTION_HUMAN_CHROM [EVEN_PREDICTION_HUMAN_CHROM ...]
		                        human chromosomes to include in even prediction data
		  --even-training-mouse-chrom EVEN_TRAINING_MOUSE_CHROM [EVEN_TRAINING_MOUSE_CHROM ...]
		                        mouse chromosomes to include in even training data
		  --even-validation-mouse-chrom EVEN_VALIDATION_MOUSE_CHROM [EVEN_VALIDATION_MOUSE_CHROM ...]
		                        mouse chromosomes to include in even validation data
		  --even-test-mouse-chrom EVEN_TEST_MOUSE_CHROM [EVEN_TEST_MOUSE_CHROM ...]
		                        mouse chromosomes to include in even test data
		  --held-out-human-chrom HELD_OUT_HUMAN_CHROM [HELD_OUT_HUMAN_CHROM ...]
		                        human chromosomes held out from training, validation,
		                        and test for model comparison
		  --held-out-mouse-chrom HELD_OUT_MOUSE_CHROM [HELD_OUT_MOUSE_CHROM ...]
		                        mouse chromosomes held out from training, validation,
		                        and test for model comparison
		
		# Example: python source/splitData.py -A data/all.h.gz -B data/all.m.gz -N
		32285361 -o data/ @example/splitData_args.txt



	The human and mouse chromosomes for the split may be specified as individual input arguments or altogether in an input file. If specifying an input file, see the example above. The file should look like [this](example/splitData_args.txt) where each set of human or mouse chromosomes grouped for a specific purpose are listed. [This table](table/SupplementaryTable2.xlsx) describes this splitting procedure in more detail.

7. Prepare the data generated above for training. This involves shuffling all training examples to randomize the order of aligning pairs (while keeping the pairs of human and mouse regions intact), shuffling them again but only with the mouse regions to generate negative examples, and sampling random pairs to generate validation and test examples. 

		source/prepareData \
		  <path to directory with output files from splitData.py> \
		  <path to output directory to store shuffled/sampled data files>			
		  
		# Example:
		source/prepareData data/ data/

	Small-sized examples of output files are provided as gzipped files [here](example/), which are used as input files for supervised training in the next step. Note that these were downsampled to have only a thousand lines in each file whereas the actual files generated are much larger.

### Step 4. Train classifiers
As specified in the data split, to make predictions for pairs of human and mouse regions with human regions coming from an odd chromosome, we train classifiers using pairs coming from even human and mouse chromosomes. To make predictions for pairs of human and mouse regions with human regions coming from an even chromosome or the X chromosome, we train classifiers using pairs of regions coming from odd human and mouse chromosomes. Therefore the training step below needs to be done twice, once with odd chromosome training data and once with even chromosome training data. As noted above, examples of input files for training a classifier are provided as gzipped files [here](example/).

1. Hyper-parameter search: 
	Train 100 neural networks, each with randomly determined combinations of hyper-parameters and trained on the same set of 1 million positive and 1 million negative training examples. 

		usage: python source/train.py [-h] [-o OUTPUT_FILENAME_PREFIX] [-k] [-v] [-t]
					      [-r NEG_DATA_RATIO] [-s SEED] [-e NUM_EPOCH] -A
					      HUMAN_TRAINING_DATA_FILENAME -B
					      MOUSE_TRAINING_DATA_FILENAME -C
					      SHUFFLED_MOUSE_TRAINING_DATA_FILENAME -D
					      HUMAN_VALIDATION_DATA_FILENAME -E
					      MOUSE_VALIDATION_DATA_FILENAME -F
					      SHUFFLED_MOUSE_VALIDATION_DATA_FILENAME
					      [-tr POSITIVE_TRAINING_DATA_SIZE]
					      [-tra TOTAL_POSITIVE_TRAINING_DATA_SIZE]
					      [-va POSITIVE_VALIDATION_DATA_SIZE]
					      [-hf NUM_HUMAN_FEATURES]
					      [-mf NUM_MOUSE_FEATURES]
					      [-hrmin HUMAN_RNASEQ_MIN]
					      [-hrmax HUMAN_RNASEQ_MAX]
					      [-mrmin MOUSE_RNASEQ_MIN]
					      [-mrmax MOUSE_RNASEQ_MAX] [-b BATCH_SIZE]
					      [-l LEARNING_RATE] [-d DROPOUT_RATE]
					      [-nl1 NUM_LAYERS_1] [-nl2 NUM_LAYERS_2]
					      [-nnh1 NUM_NEURON_HUMAN_1]
					      [-nnh2 NUM_NEURON_HUMAN_2]
					      [-nnm1 NUM_NEURON_MOUSE_1]
					      [-nnm2 NUM_NEURON_MOUSE_2] [-nn1 NUM_NEURON_1]
					      [-nn2 NUM_NEURON_2]
		
		Train a neural network
		
		optional arguments:
		  -h, --help            show this help message and exit
		  -o OUTPUT_FILENAME_PREFIX, --output-filename-prefix OUTPUT_FILENAME_PREFIX
		                        output prefix (must be specified if saving (-v))
		  -k, --random-search   if hyper-parameters should be randomly set
		  -v, --save            if the trained classifier should be saved after
		                        training
		  -t, --early-stopping  if early stopping should be allowed (stopping before
		                        the maximum number of epochs if there is no
		                        improvement in validation AUROC in three consecutive
		                        epochs)
		  -r NEG_DATA_RATIO, --neg-data-ratio NEG_DATA_RATIO
		                        weight ratio of negative samples to positive samples
		                        (default: 50)
		  -s SEED, --seed SEED  random seed (default: 1)
		  -e NUM_EPOCH, --num-epoch NUM_EPOCH
		                        maximum number of training epochs (default: 100)
		
		required arguments specifying training data:
		  -A HUMAN_TRAINING_DATA_FILENAME, --human-training-data-filename HUMAN_TRAINING_DATA_FILENAME
		                        path to human training data file
		  -B MOUSE_TRAINING_DATA_FILENAME, --mouse-training-data-filename MOUSE_TRAINING_DATA_FILENAME
		                        path to mouse positive training data file
		  -C SHUFFLED_MOUSE_TRAINING_DATA_FILENAME, --shuffled-mouse-training-data-filename SHUFFLED_MOUSE_TRAINING_DATA_FILENAME
		                        path to mouse shuffled/negative training data file
		
		required arguments specifying validation data:
		  -D HUMAN_VALIDATION_DATA_FILENAME, --human-validation-data-filename HUMAN_VALIDATION_DATA_FILENAME
		                        path to human validation data file
		  -E MOUSE_VALIDATION_DATA_FILENAME, --mouse-validation-data-filename MOUSE_VALIDATION_DATA_FILENAME
		                        path to mouse positive validation data file
		  -F SHUFFLED_MOUSE_VALIDATION_DATA_FILENAME, --shuffled-mouse-validation-data-filename SHUFFLED_MOUSE_VALIDATION_DATA_FILENAME
		                        path to mouse shuffled/negative validation data file
		
		required arguments describing feature data:
		  -tr POSITIVE_TRAINING_DATA_SIZE, --positive-training-data-size POSITIVE_TRAINING_DATA_SIZE
		                        number of samples in positive training data to *use*
		                        (default: 1000000)
		  -tra TOTAL_POSITIVE_TRAINING_DATA_SIZE, --total-positive-training-data-size TOTAL_POSITIVE_TRAINING_DATA_SIZE
		                        number of samples in total positive training data to
		                        *read*
		  -va POSITIVE_VALIDATION_DATA_SIZE, --positive-validation-data-size POSITIVE_VALIDATION_DATA_SIZE
		                        number of samples in positive validation data to use
		                        (default: 100000)
		  -hf NUM_HUMAN_FEATURES, --num-human-features NUM_HUMAN_FEATURES
		                        number of human features in input vector (default:
		                        8824)
		  -mf NUM_MOUSE_FEATURES, --num-mouse-features NUM_MOUSE_FEATURES
		                        number of mouse features in input vector (default:
		                        3113)
		  -hrmin HUMAN_RNASEQ_MIN, --human-rnaseq-min HUMAN_RNASEQ_MIN
		                        minimum expression level in human RNA-seq data
		                        (default: 8e-05)
		  -hrmax HUMAN_RNASEQ_MAX, --human-rnaseq-max HUMAN_RNASEQ_MAX
		                        maximum expression level in human RNA-seq data
		                        (default: 1.11729e06)
		  -mrmin MOUSE_RNASEQ_MIN, --mouse-rnaseq-min MOUSE_RNASEQ_MIN
		                        minimum expression level in mouse RNA-seq data
		                        (default: 0.00013)
		  -mrmax MOUSE_RNASEQ_MAX, --mouse-rnaseq-max MOUSE_RNASEQ_MAX
		                        maximum expression level in mouse RNA-seq data
		                        (default: 41195.3)
		
		optional arguments specifying hyper-parameters (ignored if random search (-k) is specified):
		  -b BATCH_SIZE, --batch-size BATCH_SIZE
		                        batch size (default: 128)
		  -l LEARNING_RATE, --learning-rate LEARNING_RATE
		                        epsilon (default: 0.1)
		  -d DROPOUT_RATE, --dropout-rate DROPOUT_RATE
		                        dropout rate (default: 0.1)
		  -nl1 NUM_LAYERS_1, --num-layers-1 NUM_LAYERS_1
		                        number of hidden layers in species-specific sub-
		                        networks (default: 1)
		  -nl2 NUM_LAYERS_2, --num-layers-2 NUM_LAYERS_2
		                        number of hidden layers in final sub-network (default:
		                        1)
		  -nnh1 NUM_NEURON_HUMAN_1, --num-neuron-human-1 NUM_NEURON_HUMAN_1
		                        number of neurons in the first hidden layer in the
		                        human-specific sub-network (default :1)
		  -nnh2 NUM_NEURON_HUMAN_2, --num-neuron-human-2 NUM_NEURON_HUMAN_2
		                        number of neurons in the second hidden layer in the
		                        human-specific sub-network (default: 0)
		  -nnm1 NUM_NEURON_MOUSE_1, --num-neuron-mouse-1 NUM_NEURON_MOUSE_1
		                        number of neurons in the first hidden layer in the
		                        mouse-specific sub-network (default: 128)
		  -nnm2 NUM_NEURON_MOUSE_2, --num-neuron-mouse-2 NUM_NEURON_MOUSE_2
		                        number of neurons in the second hidden layer in the
		                        mouse-specific sub-network (default: 0)
		  -nn1 NUM_NEURON_1, --num-neuron-1 NUM_NEURON_1
		                        number of neurons in the first hidden layer in the
		                        final sub-network (default: 256)
		  -nn2 NUM_NEURON_2, --num-neuron-2 NUM_NEURON_2
		                        number of neurons in the second hidden layer in the
		                        final sub-network (default: 0)

		                        
	Here's an example of how to use the script `train.py` to train one neural network during hyper-parameter search:
								
		# Example of training with data from odd chromosomes for hyper-parameter search:
		python source/train.py	-A data/shuf_odd_training.h.gz \
					-B data/shuf_odd_training.m.gz \
					-C data/shufx2_odd_training.m.gz \
					-D data/shuf_odd_validation.h.gz \
					-E data/shuf_odd_validation.m.gz \
					-F data/shufx2_odd_validation.m.gz \
					-tra 1000000 -tr 1000000 \ 
					-s 1 -t -k \
					> NN/output/train.py.output_odd_1.txt
		
	`-k` specifies that the hyper-parameters should be chosen randomly. [This table](table/SupplementaryTable3.xlsx) lists all the hyper-parameters and their candidate values. Both positive training data size (specified by `-tr`) and total positive training data size (specified by `-tra`) is set to 1 million since we want all 100 neural networks to be trained on the first 1 million positive samples and the first 1 million negative samples in the provided training data file. It is assumed that the number of negative samples is the same as the number of positive samples.
	
	The script prints out useful information for each epoch as the training progresses. In the example above, the last line `> NN/output/train.py.output_odd_1.txt` saves this to a file. Each line is formatted as follows: 
	- Columns 1-6: seed, negative to positive sample weight ratio, number of positive training samples, number of negative training samples, number of positive validation samples, number of negative validation samples
	- Columns 7-9: batch size, learning rate (epsilon), dropout rate
	- Columns 10-12: number of hidden layers in the human-specific sub-network, mouse-specific sub-network, and final sub-network
	- Columns 13,14: number of neurons in the first and second hidden layers in the human-specific sub-network
	- Columns 15,16: number of neurons in the first and second hidden layers in the mouse-specific sub-network
	- Columns 17,18: number of neurons in the first and second hidden layers in the final sub-network
	- Columns 19,20: current epoch and training loss
	- Columns 21-26: training MSE, training AUROC, training AUPRC, mean prediction for all training samples, mean prediction for positive training samples, mean prediction for negative training samples
	- Columns 27-32: validation MSE, validation AUROC, validation AUPRC, mean prediction for all validation samples, mean prediction for positive validation samples, mean prediciton for negative validation samples

2. 	Once 100 neural networks are trained, parse through the output generated by `train.py` (e.g. `NN/output/train.py.output_odd_1.txt`), which reports the predictive performance of the neural network at every epoch. Choose the best combination of hyper-parameters based on validation AUROC. Commands like `sort -rnk 28,28` can be used to sort the output lines by validation AUROC.
	
3. Train 100 neural networks with the best combination of hyper-parameters but given randomly sampled subsets of the available training data. Below is an example of training one neural network after hyper-parameter search for prediction:

		# Example of training with data from odd chromosomes with the best combination of hyper-parameters:
		python source/train.py	-A data/shuf_odd_training.h.gz \
					-B data/shuf_odd_training.m.gz \
					-C data/shufx2_odd_training.m.gz \
					-D data/shuf_odd_validation.h.gz \
					-E data/shuf_odd_validation.m.gz \
					-F data/shufx2_odd_validation.m.gz \
					-o NN/odd_ensemble/NN \
					-tra 2208562 -ta 1000000 \
					-b 128 -l 0.1 -d 0.2 \
					-nl1 1 -nl2 1 -nl3 2 \
					-nnh1 256 -nnh2 0 -nnm1 128 -nnm2 0 -nn1 128 -nn2 32 \
					-s 1 -t -v 
		
	`-v` specifies the trained classifier to be saved so that it can be used for prediction later. Unlike the previous example provided for hyper-parameter search, here the total positive training data size (specified by `-tra`) is larger than the positive training data size (specified as 1 million by `-tr`). This allows each of the 100 neural networks to randomly sample 1 million positive and 1 million negative samples from the entire training data, resulting in 100 neural networks with the same hyper-parameters and architecture but trained on different subsets of the data.
	
	As the training progresses, an output file with the specified prefix, seed, and hyper-parameter values listed in the name (e.g. `NN/odd_ensemble/NN_9_50_1000000_1000000_100000_100000_64_0.1_0.2_1_1_1_256_0_128_0_256_0.pt` was trained with seed 9) will be updated. Once the training is over, this file would store the classifier from the epoch with the best predictive performance measured using validation data.

### Step 5. Generate genome-wide prediction
As with training, the prediction step needs to be done twice, with classifiers trained on odd chromosome data and separately with classifiers trained on even chromosome data. 

1. Use each of the 100 trained neural networks to generate predictions for pairs of human and mouse regions that were held out from training and validation.

		usage: python source/predict.py [-h] [-s SEED] [-b BATCH_SIZE] -t
		                                TRAINED_CLASSIFIER_FILENAME -H
		                                HUMAN_FEATURE_FILENAME -M
		                                MOUSE_FEATURE_FILENAME -d DATA_SIZE -o
		                                OUTPUT_FILENAME [-hf NUM_HUMAN_FEATURES]
		                                [-mf NUM_MOUSE_FEATURES]
		                                [-hrmin HUMAN_RNASEQ_MIN]
		                                [-hrmax HUMAN_RNASEQ_MAX]
		                                [-mrmin MOUSE_RNASEQ_MIN]
		                                [-mrmax MOUSE_RNASEQ_MAX]
		
		Generate predictions given a trained neural network
		
		optional arguments:
		  -h, --help            show this help message and exit
		  -s SEED, --seed SEED  random seed (default: 1)
		  -b BATCH_SIZE, --batch-size BATCH_SIZE
		                        batch size (default: 128)
		
		required arguments specifying input and output:
		  -t TRAINED_CLASSIFIER_FILENAME, --trained-classifier-filename TRAINED_CLASSIFIER_FILENAME
		                        path to a trained classifier (.pt)
		  -H HUMAN_FEATURE_FILENAME, --human-feature-filename HUMAN_FEATURE_FILENAME
		                        path to human feature data file
		  -M MOUSE_FEATURE_FILENAME, --mouse-feature-filename MOUSE_FEATURE_FILENAME
		                        path to mouse feature data file
		  -d DATA_SIZE, --data-size DATA_SIZE
		                        number of samples
		  -o OUTPUT_FILENAME, --output-filename OUTPUT_FILENAME
		                        path to output file
		  -hf NUM_HUMAN_FEATURES, --num-human-features NUM_HUMAN_FEATURES
		                        number of human features in input vector (default:
		                        8824)
		  -mf NUM_MOUSE_FEATURES, --num-mouse-features NUM_MOUSE_FEATURES
		                        number of mouse features in input vector (default:
		                        3113)
		  -hrmin HUMAN_RNASEQ_MIN, --human-rnaseq-min HUMAN_RNASEQ_MIN
		                        minimum expression level in human RNA-seq data
		                        (default: 8e-05)
		  -hrmax HUMAN_RNASEQ_MAX, --human-rnaseq-max HUMAN_RNASEQ_MAX
		                        maximum expression level in human RNA-seq data
		                        (default: 1.11729e06)
		  -mrmin MOUSE_RNASEQ_MIN, --mouse-rnaseq-min MOUSE_RNASEQ_MIN
		                        minimum expression level in mouse RNA-seq data
		                        (default: 0.00013)
		  -mrmax MOUSE_RNASEQ_MAX, --mouse-rnaseq-max MOUSE_RNASEQ_MAX
		                        maximum expression level in mouse RNA-seq data
		                        (default: 41195.3)
		
		# Example: python source/predict.py -t NN/odd_ensemble/NN_1_*.pt -H
		data/even_all.h.gz -M data/even_all.m.gz -d 16627449 -o
		NN/odd_ensemble/even_all_1.gz

	The number of features and RNA-seq minimum and maximum should be the same as the ones provided to `train.py`.
			
2. For each pair of human and mouse regions, average the 100 predictions, each generated by one neural network, to generate the final human-mouse LECIF score.

		usage: python source/generateBrowserTrack.py [-h] -p REGION_FILENAME -f
		                                             SCORE_FILENAMES
		                                             [SCORE_FILENAMES ...] -o
		                                             OUTPUT_FILENAME
		                                             [-a INDEX_TO_USE_FOR_CHROMOSOME]
		                                             [-b INDEX_TO_USE_FOR_START]
		                                             [-c INDEX_TO_USE_FOR_END]
		
		Generate a genome browser track given multiple predictions generated by
		predict.py
		
		optional arguments:
		  -h, --help            show this help message and exit
		  -a INDEX_TO_USE_FOR_CHROMOSOME, --index-to-use-for-chromosome INDEX_TO_USE_FOR_CHROMOSOME
		                        column index to use for human chromosome in the
		                        specified region file
		  -b INDEX_TO_USE_FOR_START, --index-to-use-for-start INDEX_TO_USE_FOR_START
		                        column index to use for human region start position in
		                        the specified region file
		  -c INDEX_TO_USE_FOR_END, --index-to-use-for-end INDEX_TO_USE_FOR_END
		                        column index to use for human region end position in
		                        the specified region file
		
		required arguments specifying input and output:
		  -p REGION_FILENAME, --region-filename REGION_FILENAME
		                        path to the output file (.gz) from samplePairs.py
		  -f SCORE_FILENAMES [SCORE_FILENAMES ...], --score-filenames SCORE_FILENAMES [SCORE_FILENAMES ...]
		                        list of output files from predict.py
		  -o OUTPUT_FILENAME, --output-filename OUTPUT_FILENAME
		                        path to final output file
		
		# Example: python source/generateBrowserTrack.py -p
		position/hg19.evenchr.mm10.50bp.gz -f NN/odd_ensemble/even_all_*.gz -o
		NN/odd_ensemble/even_all.gz
		
	This script can also be used to generate scores for positive and negative test data and held-out data for model comparison. 
										
3. Convert the resulting Bed file into a BigWig file using [bedGraphToBigWig](http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64.v369/). This may be done before or after predictions for odd and even chromosomes are concatenated.

## Applying LECIF to other pairs of species
LECIF can be applied to any pair of species, though the quality of the score will depend on the coverage of the data available for both species. The steps described above may need to be modified accordingly.

## Authors
Soo Bin Kwon (University of California, Los Angeles), Jason Ernst (University of California, Los Angeles).

## Acknowledgements
We thank Trevor Ridgley (University of California, Santa Cruz) and Grace Casarez (University of California, Santa Barbara) for their contribution during the Bruins-In-Genomics (B.I.G.) Summer Research Program.

## Reference
Kwon, S.B., Ernst, J. Learning a genome-wide score of human–mouse conservation at the functional genomics level. Nat Commun 12, 2495 (2021). https://doi.org/10.1038/s41467-021-22653-8
