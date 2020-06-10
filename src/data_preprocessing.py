'''
I used the pandas-plink library to load the data out of bed, bim, and fam files. I needed all the data to be
in a tabular format, but this pandas-plink does not provide an easy mechanism to get that. So, I had to go over
the each sample (row) and retrieve the mutations' (columns) information iteratively. Because the retrieval for
each row was very time consuming, reading the whole data this way could take hours (or days). So, to make it
faster to load the data for future use, once I read the data for a row (as a numpy array), I saved the numpy
array as a pickle file on the disk. So, for future use, all I need was to read those pickle file and construct
my final numpy array. Reading the pickle file is very fast, and the whole data can be loaded in few minutes
'''
from pandas_plink import read_plink1_bin
import os

testing = TRUE
if testing:
    from os.path import join
    from pandas_plink import get_data_folder
        G = read_plink1_bin(join(get_data_folder(), "chr*.bed"), verbose=False)
        print(G)
        #(bim, fam, bed) = read_plink(join(get_data_folder(), "data"), verbose=False)
        #print(bed.compute())
        #SNPS genotype dataset
        data_GT =  G.compute().values #samples x snps
        samples =  G.sample.values  # samples
        snps = G.snp.values #snps
        #Identify some causal snps (known)
else:
    #datapath = "Z:\\POPi3\\Staff\\Raquel Aoki\\Cisplatin-induced_ototoxicity"
    #os.chdir(datapath)
    G = read_plink1_bin("Peds_CIO_merged_qc_data.bed", "Peds_CIO_merged_qc_data.bim", "Peds_CIO_merged_qc_data.fam", verbose=False)
