# Summer2020MultipleCauses
BCCH project 

Todo: 
1) Add the average run time
2) Run again for the new clinical variables



Encoding: 
Values available: 0 (aa) , 1(Aa) and 2 (AA) 

Dominant: 0(aa) -> 1; 1(Aa) and 2 (AA) -> 0
Recessive: 2(AA) -> 1; 0(aa) and 1(Aa) -> 0 

I work with python, so the files were saved using python format (more efficient).
Here are the commands I used: 

sparse.save_npz(path_output+tag+'gt_dominant.npz',data_df)
sparse.save_npz(path_output+tag+'gt_recessive.npz',data_sf)
np.save(path_output+tag + 'variants',variants_f)
np.save(path_output+tag + 'cadd',cadd_f)
