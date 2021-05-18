if 'bcch' in params['data']:
    data = goPDX(final=True)
    from sklearn.preprocessing import MinMaxScaler

    # cleaning clinical names
    drop = ['CisplatinExactDuration (days)', 'Carboplatin (1=yes,0=no)',
            'CarboplatinExactDuration (days)', 'TumorType',
            'Vancomycin_Concomitant (1=yes,0=no)',
            'Tobramycin_Concomitant (1=yes,0=no)',
            'Gentamicin_Concomitant (1=yes,0=no)',
            'Amikacin_Concomitant (1=yes,0=no)',
            'Furosemide_Concomitant (1=yes,0=no)',
            'Otoprotectant_Given',
            'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9',
            'PC10', 'Array', 'Cranial_Irradiation (1=yes,0=no)',
            'AmikacinConcomitantDuration (days)']

    keep_index = []
    for i, colname in enumerate(data.x_clinical_names):
        if colname not in drop:
            keep_index.append(i)

    data.x_clinical_names = data.x_clinical_names[keep_index]
    data.x_clinical = data.x_clinical[:, keep_index]

    s01 = MinMaxScaler()
    data.x_clinical = s01.fit_transform(data.x_clinical)

    treatment_names = ['CisplatinDose_cumulative (mg/m2)',
                       'CarboplatinDose_cumulative(mg/m2)',
                       'HeadAndNeckRadiation (1=yes,0=no)',
                       'VancomycinConcomitantDuration (days)',
                       'TobramycinConcomitantDuration (days)',
                       'GentamicinConcomitantDuration (days)',
                       'FurosemideConcomitantDuration (days)',
                       'Vincristine_Concomitant (1=yes,0=no)',
                       ]

    treatement_columns = []
    for i, colname in enumerate(data.x_clinical_names):
        if colname in treatment_names:
            treatement_columns.append(i)

    y01 = data.y
    print(data.x_clinical.shape, data.x_snps.shape)

    X1_cols = list(range(data.x_clinical.shape[1] - len(treatement_columns)))
    X2_cols = list(range(data.x_clinical.shape[1] - len(treatement_columns),
                         data.x_clinical.shape[1] - len(treatement_columns) + data.x_snps.shape[1]))

    X = np.concatenate((data.x_clinical, data.x_snps), 1)
    X = pd.DataFrame(X)
    treatment_effects = np.repeat(0, len(treatement_columns))
    params_b = {'DA': {'k': [5]},
                'CEVAE': {'num_epochs': 150, 'batch': 10, 'z_dim': 5, 'binarytarget': False},
                'Dragonnet': {'u1': 200, 'u2': 100, 'u3': 1}}  # same as paper

    if params['baselines']:
        baselines_results, exp_time, f1_test = baselines(params['baselines_list'], X, y01, params_b,
                                                         TreatCols=treatement_columns, timeit=True,
                                                         seed=seed_models)
    else:
        baselines_results, exp_time, f1_test = baselines(['noise'], X, y01, params_b,
                                                         TreatCols=treatement_columns, timeit=True,
                                                         seed=seed_models)
    start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y01, test_size=0.33, random_state=seed_models)

    data_nnl = m3e2.data_nn(X_train.values, X_test.values, y_train, y_test, treatement_columns,
                            treatment_effects, X1_cols, X2_cols)
    loader_train, loader_val, loader_test, num_features = data_nnl.loader(params['suffle'], params['batch_size'],
                                                                          seed_models)
    params['hidden1'] = trykey(params, 'hidden1', 6)
    params['hidden2'] = trykey(params, 'hidden2', 6)
    params['pos_weights'] = np.repeat(params['pos_weights'], len(treatement_columns))
    cate_m3e2, f1_test_ = m3e2.fit_nn(loader_train, loader_val, loader_test, params, treatement_columns,
                                      num_features, X1_cols=X1_cols, X2_cols=X2_cols, use_bias_y=True)
    print('... CATE')
    baselines_results['M3E2'] = cate_m3e2
    exp_time['M3E2'] = time.time() - start_time
    f1_test['M3E2'] = f1_test_
    output = organize_output(baselines_results.copy(), treatment_effects,
                             exp_time, f1_test, gwas=False)