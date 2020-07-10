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


path_input = "Z:\\POPi3\\Staff\\Raquel Aoki\\Cisplatin-induced_ototoxicity"
path_output = "Z:\\POPi3\\Staff\\Raquel Aoki\\Summer2020\\"
path_output_simulated = "C:\\Users\\raque\\Documents\\GitHub\\Summer2020MultipleCauses\\data\\"


def ld_prune(gn,variants, size, step, threshold=.1, n_iter=1):
    #https://en.wikipedia.org/wiki/Linkage_disequilibrium
    for i in range(n_iter):
        loc_unlinked = allel.locate_unlinked(gn, size=size, step=step, threshold=threshold)
        n = np.count_nonzero(loc_unlinked)
        n_remove = gn.shape[0] - n
        #print('iteration', i+1, 'retaining', n, 'removing', n_remove, 'variants')
        gn = gn.compress(loc_unlinked, axis=0)
        variants = variants.compress(loc_unlinked)
    return gn, variants

#Couple of hours
def data_in_npz(path_input, path_output):
    os.chdir(path_input)
    G = read_plink1_bin("Peds_CIO_merged_qc_data.bed", bim = None, fam = None, verbose=False)
    #data_GT =  G.compute().values #samples x snps #memory problem
    #read parts of bed values https://github.com/limix/pandas-plink/blob/master/doc/usage.rst
    samples =  G.sample.values  # samples
    variants = G.variant.values

    print('Shape: ',len(samples),len(variants)) #Shape:  454 6726287

    #print(G.a0.sel(variant=str(i)).values,G.a1.sel(variant=str(i)).values)
    #print(G.sel(sample=samples[0], variant=variants[0]).values)
    #https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas

    #saving samples
    np.save(path_output + 'samples',samples)

    #preparing variables
    variants_done = []
    start0 = start = time.time()
    interval = 100000 #1000
    ind = 0
    build = True

    for var in range(len(variants)//interval):
        row = G.sel(sample=samples, variant=variants[ind:ind+interval]).values.transpose()
        row_,variants_  = ld_prune(row,variants[ind:ind+interval], 600 , 200, 0.1, 3)

        variants_done.append(variants_)
        ind = ind + interval


        if build:
            data = np.matrix(row_)
            build = False
        else:
            data = np.concatenate((data,np.matrix(row_)), axis = 0)

        if var%10 == 0:
            print('Progress: ',round(var*100/(len(variants)//interval),4),'% ---- Time Dif (s): ', round(time.time()-start,2))
            np.savez(path_output+'gt.npz', name1=data)
            start = time.time()


    row = G.sel(sample=samples, variant=variants[ind:len(variants)]).values.transpose()
    row_,variants_  = ld_prune(row,variants[ind:ind+interval], 600 , 200, 0.1, 3)

    variants_done.append(variants_)
    data = np.concatenate((data,np.matrix(row_)), axis = 0)
    np.savez(path_output+'gt.npz', name1=data)
    np.save(path_output + 'variants',variants_)
    print('Total time in minutes: ', round((time.time()-start0)/60, 2))



    #last prune variants = variants.compress(loc_unlinked)
    #IndexError: index 997 is out of bounds for axis 0 with size 997
    data, variants_ = ld_prune(data,variants_,600 , 200, 0.1, 5)
    np.savez(path_output+'gt.npz', name1=data)
    np.save(path_output + 'variants',variants_)


    #add back known SNPs



    #SNPS 01
    data = np.where(data,0,1).transpose()
    sumcol = data.sum(axis = 0)
    data = data.compress(sumcol!=0, axis=0)
    variants_ = variants_.compress(sumcol!=0)

    sparse.save_npz(path_output+'gt01.npz', data.transpose())
    np.save(path_output + 'variants01',variants_)

    return data, variants_


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
