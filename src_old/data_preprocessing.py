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
from scipy.sparse import coo_matrix, vstack
from scipy import sparse

path_input = "Z:\\POPi3\\Staff\\Raquel Aoki\\Cisplatin-induced_ototoxicity"
path_output = "Z:\\POPi3\\Staff\\Raquel Aoki\\Summer2020\\"
path_output_simulated = "C:\\Users\\raque\\Documents\\GitHub\\Summer2020MultipleCauses\\data\\"

#ks = pd.read_csv(path_input+'\\SNPS_known.txt', header = None)[0].values
known_snps_path = 'C:\\Users\\raque\\Documents\\GitHub\\Summer2020MultipleCauses\\data\\known_snps.txt'
ks = pd.read_csv(known_snps_path, header = None)[0].values

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
    for i in range(len(ks)):
        ksf = cadd.variants[cadd.ID==ks[i]]
        if len(ksf)==0: 
            missing.append(ks[i])              
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
    correlations = pd.DataFrame(correlations)
    correlations.fillna(1,inplace = True)
    correlations = correlations.values
    del r 
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
                #if all less than 1, keep none
                filter_2 = np.equal(v_cadd,v_cadd.max())
                if v_cadd.max()>1: 
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
    #ADD FUNCTION TO KEEP KNOWN ELEMENTS HERE
    
    #keep = [item for sublist in keep for item in sublist]    
    loc_unlinked = np.zeros(len(variants))
    loc_unlinked[keep] = 1
    
    #n = np.count_nonzero(loc_unlinked)
    #n_remove = gn.shape[0] - n
    #print('retaining', n, 'removing', n_remove, 'variants')
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


def filteringSNPS(variants, cadd_sort, samples, G, 
                  path_input, path_output, tag, 
                  thold = 0.05, interval = 10000): 
    '''preparing variables'''
    variants_done = []
    cadd_done = []
    start0 = start = time.time()
    #interval = 10000
    ind = 0
    #thold = 0.05
    build = True

    '''Filtering full dataset'''
    print(range(len(variants)//interval))
    for var in range(len(variants)//interval):
    #use sparse format to reduce size
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.vstack.html
    #for var in range(11):
    
        row = G.sel(sample=samples, variant=variants[ind:ind+interval]).values.transpose()
        #Counting Frequencies
        
        row = pd.DataFrame(row)
        row.fillna(-1, inplace=True)
        #counts = row.copy() 
        #counts_ = counts.apply(pd.Series.value_counts , axis = 1)
        #del counts 
        
        row = row.values
        row_,variants_ , cadd_ = ld_prune(row, variants[ind:ind+interval], cadd_sort[ind:ind+interval], thold) #600 , 200, 0.05, 3
        del row  
        
        variants_done.append(variants_)
        cadd_done.append(cadd_)
        ind = ind + interval
        row_d = np.where(row_,0,1) #Dominant: 0 are 1; 1 and 2 are 0 
        row_s = np.where(row_,2,1) #Recessive: 2 are 1, 1 and 0 are 0 #WRONG
        #Correct filter for Resessive:np.where(test > 1, 1, 0)
    
        if build:
            #dominant coding: 
            data_d = coo_matrix(row_d)
            
            #recessive coding:
            data_s = coo_matrix(row_s)
            ##  data = np.matrix(row_)
            #counts012 = counts_.values
            build = False
            del row_d, row_s, row_ 
        else:
            data_d = vstack([data_d,coo_matrix(row_d)])
            data_s = vstack([data_s,coo_matrix(row_s)])
            #data = np.concatenate((data,np.matrix(row_)), axis = 0)
            #data = vstack([data,coo_matrix(row_)])#.toarray()
            #counts012 = np.concatenate((counts012, counts_.values), axis = 0) 
    
        if var%10 == 0:
            print('Progress: ',round(var*100/(len(variants)//interval),4),'% ---- Time Dif (s): ', round(time.time()-start,2))
            #np.savez(path_output+'gt.npz', name1=data)
            sparse.save_npz(path_output+tag+'gt_dominant.npz',data_d)
            sparse.save_npz(path_output+tag+'gt_recessive.npz',data_s)
            np.save(path_output+tag + 'variants',variants_done)
            np.save(path_output+tag + 'cadd',cadd_done)
            #np.save(path_output + 'counts012',counts012)
            start = time.time()
    
    
    row = G.sel(sample=samples, variant=variants[ind:len(variants)]).values.transpose()
    row = pd.DataFrame(row)
    row.fillna(-1, inplace=True)
    #counts = row.copy() 
    #counts_ = counts.apply(pd.Series.value_counts , axis = 1)
    #del counts 
    
    row = row.values
    row_,variants_ , cadd_ = ld_prune(row, variants[ind:ind+interval], cadd_sort[ind:ind+interval], thold) #600 , 200, 0.05, 3
    variants_done.append(variants_)
    cadd_done.append(cadd_)
    
    row_d = np.where(row_,0,1)
    row_s = np.where(row_,2,1)   
    del row_
    
    #data = np.concatenate((data,np.matrix(row_)), axis = 0)
    #counts012 = np.concatenate((counts012, counts_.values), axis = 0) 
    data_d = vstack([data_d,coo_matrix(row_d)])
    data_s = vstack([data_s,coo_matrix(row_s)])
            
    variants_done = [item for sublist in variants_done for item in sublist]    
    cadd_done = [item for sublist in cadd_done for item in sublist]    
    
    sparse.save_npz(path_output+tag+'gt_dominant.npz',data_d)
    sparse.save_npz(path_output+tag+'gt_recessive.npz',data_s)
    np.save(path_output+tag + 'variants',variants_done)
    np.save(path_output+tag + 'cadd',cadd_done)
    #np.save(path_output + 'counts012',counts012)

    print('Total time in minutes: ', round((time.time()-start0)/60, 2))
    return data_d, data_s, variants_done, cadd_done



'''First PRUNE: IF 0 IN ONE AND 1 ON ANOTHER: 2, IF 1 AND 0: 0, IF 0 AND 0: 1'''
#Takes 48 hours to finish 
data_d, data_s, variants_, cadd_ = filteringSNPS(variants, cadd_sort.CADD_PHRED.values, samples, G, path_input, path_output, 'first')


'''FINAL PRUNE: IF 0 IN ONE AND 1 ON ANOTHER: 2, IF 1 AND 0: 0, IF 0 AND 0: 1'''
data_df, data_sf, variants_f, cadd_f = filteringSNPS(np.array(variants_), np.array(cadd_), samples, G, path_input, path_output, 'final', 0.2, 10000)


'''Fixing Errors'''
tag = 'snpsback_'
data_df = sparse.load_npz(path_output+tag+'gt_dominant.npz')
data_sf = sparse.load_npz(path_output+tag+'gt_recessive.npz')
variants_f = np.load(path_output+tag + 'variants.npy')
cadd_f = np.load(path_output+tag+'cadd.npy')

missing = variants_f[data_df.shape[0]:len(variants_f)]
row_ = G.sel(sample = samples, variant = missing).values.transpose()
row_d = np.where(row_,0,1)
data_df = vstack([data_df,coo_matrix(row_d)])
        
row_ = G.sel(sample = samples, variant = variants_f).values.tranpose()
row_d = np.equal(row_,0)
row_d = row_d*1

row_r = np.equal(row_,2)
row_r = row_r*1

data_df = coo_matrix(row_d)
data_sf = coo_matrix(row_r)
sparse.save_npz(path_output+tag+'gt_dominant.npz',data_df)
sparse.save_npz(path_output+tag+'gt_recessive.npz',data_sf)

'''Adding back SNPS'''

for ks in ks_fullname: 
    if ks not in variants_f: 
        #print(ks)
        row_ = G.sel(sample=samples, variant=ks).values.transpose()
        row_d = np.where(row_,0,1)
        row_s = np.where(row_,2,1)   
        
        data_df = vstack([data_df,coo_matrix(row_d)])
        data_sf = vstack([data_sf,coo_matrix(row_s)])
    
        #data = np.concatenate((data,np.matrix(row_)), axis = 0)
        
        variants_f.append(ks)
        cadd_f.append(cadd_sort.CADD_PHRED[cadd_sort.variants==ks].values[0])

tag = 'snpsback2_'
sparse.save_npz(path_output+tag+'gt_dominant.npz',data_df)
sparse.save_npz(path_output+tag+'gt_recessive.npz',data_sf)
np.save(path_output+tag + 'variants',variants_f)
np.save(path_output+tag + 'cadd',cadd_f)

