clear

train_behavior_xlsx = "bug_ev1/behavior_data1.xlsx";
test_behavior_xlsx = "bug_ev1/behavior_data5.xlsx";
zscore_mat = "bug_ev1/df_f_zscore.npy.mat";
event1_length = 3299;
event234_length = 5749;

[train_AUC, test_AUC] = LDA_anal_ADASYN_sxl(train_behavior_xlsx, test_behavior_xlsx, zscore_mat, event1_length, event234_length, 0);

fprintf("train auc is %f\n", train_AUC);
fprintf("test auc is %f\n", test_AUC);

[train_AUC, test_AUC_random] = LDA_anal_ADASYN_sxl(train_behavior_xlsx, test_behavior_xlsx, zscore_mat, event1_length, event234_length, 500);

pvalue = sum(test_AUC < test_AUC_random) / length(test_AUC_random);
fprintf("p-value is %f", pvalue);
