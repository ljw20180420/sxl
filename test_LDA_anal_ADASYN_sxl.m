[train_AUC, test_AUC] = LDA_anal_ADASYN_sxl("for_LJW/F2_2/behavior3.xlsx", "for_LJW/F2_2/behavior5.xlsx", "for_LJW/F2_2/df_f_zscore.npy.mat", 6299, 5749, 0);

[train_AUC, test_AUC_random] = LDA_anal_ADASYN_sxl("for_LJW/F2_2/behavior3.xlsx", "for_LJW/F2_2/behavior5.xlsx", "for_LJW/F2_2/df_f_zscore.npy.mat", 6299, 5749, 500);

disp(train_AUC)
disp(test_AUC)
disp(test_AUC_random)
