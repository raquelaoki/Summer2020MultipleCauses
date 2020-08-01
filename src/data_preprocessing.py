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
import numpy as np 
import pandas as pd 
import time
import allel
from scipy.spatial.distance import squareform
import operator
import functools 

path_input = "Z:\\POPi3\\Staff\\Raquel Aoki\\Cisplatin-induced_ototoxicity"
path_output = "Z:\\POPi3\\Staff\\Raquel Aoki\\Summer2020\\"
path_output_simulated = "C:\\Users\\raque\\Documents\\GitHub\\Summer2020MultipleCauses\\data\\"

known_snps_path = "C:\\Users\\raque\\Documents\\GitHub\\Summer2020MultipleCauses\\data\\known_snps.txt"
ks = pd.read_csv(known_snps_path, header = None)

def functools_reduce_iconcat(a):
    return functools.reduce(operator.iconcat, a, [])


def read_cadd(path_input, ks): 
    print('LOADING CADD')
    os.chdir(path_input)
    cadd = pd.read_csv('Peds_CIO_merged_qc_CADD.txt', sep = " ")
    cadd['variants'] = cadd['#CHROM'].astype(str)+"_"+cadd['ID']
    cadd0 = cadd.iloc[:,[6,5]]
    
    print('CHECKING FOR KNOWN SNPS')
    ks_fullname = []
    missing = []
    for i in range(len(ks[0])):
        ksf = cadd.variants[cadd.ID==ks[0][i]]
        if len(ksf)==0: 
            missing.append(ks[0][i])
        else: 
            ks_fullname.append(ksf.values[0])
    
       
    
    return cadd0, ks_fullname, missing


def ld_prune(gn,variants,cadd,thold):
    
    '''
    input: 
        subset of the gn, variants and cadd associated to this subset
    output: 
        subset of the input subset without high correlated snps and cadd above 1
    
    '''    
    #https://en.wikipedia.org/wiki/Linkage_disequilibrium
    #Estimate the linkage disequilibrium parameter r for each pair of variants 
    r = allel.rogers_huff_r(gn)
    correlations = squareform(r ** 2)
    
    #Saving the indiced of explored snps 
    keep = []
    done = []
    
    for v_ in range(len(variants)):
        if v_ not in done:
            #Filtering out explored columns
            nextcolumns = set(np.arange(len(variants)))-set(done)
            filter_0 = np.zeros(len(variants))
            filter_0[list(nextcolumns)] = 1
            
            #Filtering the columns with high correlation
            filter_1 = np.greater(correlations[:,v_], thold)
            filter_1 = filter_1*np.equal(filter_0,1)
    
    
            if filter_1.sum()>1:
                v_ind = np.arange(len(variants))[filter_1]  
                v_ind = np.append(v_ind, v_) 
                
                v_cadd = cadd[filter_1]
                v_cadd = np.append(v_cadd, cadd[v_])
                       
                #keeping only the snp with highest cadd
                filter_2 = np.equal(v_cadd,v_cadd.max())
                if isinstance(v_ind[filter_2], np.ndarray): 
                    keep.append(v_ind[filter_2][0])
                else: 
                    keep.append(v_ind[filter_2])
                  
                
                for item in v_ind: 
                    done.append(item)
            else: 
                keep.append(v_)
                done.append(v_)
        
    
    #Filtering final results on the subset to output
    
    #keep = [item for sublist in keep for item in sublist]    
    loc_unlinked = np.zeros(len(variants))
    loc_unlinked[keep] = 1
    
    n = np.count_nonzero(loc_unlinked)
    n_remove = gn.shape[0] - n
    print('retaining', n, 'removing', n_remove, 'variants')
    gn = gn.compress(loc_unlinked, axis=0)
    variants = variants[keep]
    cadd = cadd[keep]
    return gn, variants,cadd



'''Loading Files'''
os.chdir(path_input)  
G = read_plink1_bin("Peds_CIO_merged_qc_data.bed", bim = None, fam = None, verbose=False)
#data_GT =  G.compute().values #samples x snps #memory problem
#read parts of bed values https://github.com/limix/pandas-plink/blob/master/doc/usage.rst
#https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas

samples =  G.sample.values  # samples
variants = G.variant.values
s, v = len(samples), len(variants)
print('Original shape: ',len(samples),len(variants)) #Shape:  454 6726287

cadd, ks_fullname, missing = read_cadd(path_input,ks)


'''Saving samples output'''
np.save(path_output + 'samples',samples)

'''preparing variables'''
variants_done = []
cadd_done = []
start0 = start = time.time()
interval = 10000
ind = 0
thold = 0.05
build = True

'''Making sure the Cadd and variants are in the same order (very important)'''
cadd['variants_cat'] = pd.Categorical(cadd['variants'], categories=variants,   ordered=True)
cadd_sort = cadd.sort_values(by=['variants_cat'])
cadd_sort.reset_index(inplace=True)
if np.equal(cadd_sort.variants,variants).sum()==len(variants):
    print('CADD and variants are in the same order')
    del cadd
else: 
    print('ERROR: CADD and variantres are in DIFFERENT order')

cadd_sort.fillna(value = {'CADD_PHRED':0}, inplace = True)

'''Filtering full dataset'''
#for var in range(len(variants)//interval):
for var in range(10):

    row = G.sel(sample=samples, variant=variants[ind:ind+interval]).values.transpose()
    row_,variants_ , cadd_ = ld_prune(row, variants[ind:ind+interval], cadd_sort.CADD_PHRED[ind:ind+interval].values, thold) #600 , 200, 0.05, 3
     
    variants_done.append(variants_)
    cadd_done.append(cadd_)
    ind = ind + interval

    if build:
        data = np.matrix(row_)
        build = False
    else:
        data = np.concatenate((data,np.matrix(row_)), axis = 0)

    if var%10 == 0:
        print('Progress: ',round(var*100/(len(variants)//interval),4),'% ---- Time Dif (s): ', round(time.time()-start,2))
        np.savez(path_output+'gt.npz', name1=data)
        np.save(path_output + 'variants',variants_done)
        np.save(path_output + 'cadd',cadd_done)
        start = time.time()


row = G.sel(sample=samples, variant=variants[ind:len(variants)]).values.transpose()
row_,variants_ , cadd_ = ld_prune(row, variants[ind:ind+interval], cadd_sort.CADD_PHRED[ind:ind+interval].values, thold) #600 , 200, 0.05, 3
variants_done.append(variants_)
cadd_done.append(cadd_)

data = np.concatenate((data,np.matrix(row_)), axis = 0)
variants_done = [item for sublist in variants_done for item in sublist]    
cadd_done = [item for sublist in cadd_done for item in sublist]    

#last prune variants = variants.compress(loc_unlinked)
#IndexError: index 997 is out of bounds for axis 0 with size 997
data_,variants_ , cadd_ = ld_prune(data,  np.asarray(variants_done),  np.asarray(cadd_done), thold) #600 , 200, 0.05, 3
np.savez(path_output+'gt.npz', name1=data_)
np.save(path_output + 'variants',variants_)
np.save(path_output + 'cadd',cadd_)

print('Total time in minutes: ', round((time.time()-start0)/60, 2))

'''Adding back SNPS'''
for ks in ks_fullname: 
    if ks not in variants_: 
        #Recover position in G.sel 
        #add back on data
        #add on varians 
        #add on cadd


#add back known SNPs
#use both encodings
#dominant coding (0-> 1 and 1, 2 -> 0) to have recessive coding (0,1-> 1 and 2 -> 0) 

#dominant coding: 
data_d = np.where(data_,0,1).transpose()
sparse.save_npz(path_output+'gt_dominant.npz',data_d)

#recessive coding: 
data_s = np.where(data_,2,1).transpose()
sparse.save_npz(path_output+'gt_recessive.npz',data_d)


#return data, variants_
#end function

load = False
if load:
    bd = np.load(path_output+'gt.npz')
    bd = bd['name1']

else:
    bd, variants = data_in_npz(path_input, path_output)





#sparse.save_npz(path_output+'genotype_sparse.npz', data)
#sparse_matrix = scipy.sparse.load_npz('/tmp/sparse_matrix.npz')


#sparse_matrix =
#bd.load
#bd.todense()


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
