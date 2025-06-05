function [train_AUC, test_AUC] = LDA_anal_ADASYN_sxl(train_behavior_xlsx, test_behavior_xlsx, zscore_mat, event1_length, event234_length, perm)
    % Calculating LDA performance level 
    % This function includes the ADASYN method as proposed in the
    % following paper
    %[1]: H. He, Y. Bai, E.A. Garcia, and S. Li, "ADASYN: Adaptive Synthetic
    %Sampling Approach for Imbalanced Learning", Proc. Int'l. J. Conf. Neural
    %Networks, pp. 1322--1328, (2008).

    % Run this function in Example_data\#512 folder to test this function. 

    %-----------------
    % Date: 2024-04-30
    % Author: Choi et al.
    %-----------------

    function range = event_range(start_name, event1_length, event234_length, nframe)
        if lower(start_name) == "start1"
            range = 1:event1_length;
            return;
        end
        if lower(start_name) == "start5"
            range = event1_length + event234_length + 1 : nframe;
            return;
        end
        range = event1_length + 1 : event1_length + event234_length;
    end

    % load zscore, the name is df_f_zscore
    load(zscore_mat, 'df_f_zscore');
    nframe = size(df_f_zscore, 2);
    % load train_behavior
    train_behavior = readtable(train_behavior_xlsx);
    train_labels = zeros(1, nframe);
    for row = 1:height(train_behavior)
        train_labels(train_behavior{row, 1} : train_behavior{row, 2}) = 1;
    end
    train_range = event_range(train_behavior.Properties.VariableNames{1}, event1_length, event234_length, nframe);
    train_labels = train_labels(train_range);
    train_zscore = df_f_zscore(:, train_range);
    % load test_behavior
    test_behavior = readtable(test_behavior_xlsx);
    test_labels = zeros(1, nframe);
    for row = 1:height(test_behavior)
        test_labels(test_behavior{row, 1} : test_behavior{row, 2}) = 1;
    end
    test_range = event_range(test_behavior.Properties.VariableNames{1}, event1_length, event234_length, nframe);
    test_labels = test_labels(test_range);
    test_zscore = df_f_zscore(:, test_range);
        
    if sum(train_labels) ~= 0 
        %%%%%% ADASYN 
        adasyn_features                 = train_zscore;
        adasyn_labels                   = train_labels;
        adasyn_beta                     = [];   %let ADASYN choose default
        adasyn_kDensity                 = [];   %let ADASYN choose default
        adasyn_kSMOTE                   = [];   %let ADASYN choose default
        adasyn_featuresAreNormalized    = false;    %false lets ADASYN handle normalization
        
        [adasyn_featuresSyn, adasyn_labelsSyn] = ADASYN(adasyn_features', adasyn_labels, adasyn_beta, adasyn_kDensity, adasyn_kSMOTE, adasyn_featuresAreNormalized);
        X_train_balanced = [adasyn_features'; adasyn_featuresSyn];
        X_label_train_balanced = [adasyn_labels, adasyn_labelsSyn'];


        Var =transpose([X_train_balanced'; X_label_train_balanced]);
        cell_rem = [];
        for i = 1:size(Var,2)-1
            if floor(Var(1,i)*10000) == floor(Var(end,i)*10000)
                cell_rem(length(cell_rem) + 1) = i;
            end
        end
        Var(:, cell_rem) = [];                         
        training_OFL1= array2table(Var);

        % linear discriminant analysis using MATLAB function 
        [trainedModel, validationAccuracy] = combined_trainClassifier(training_OFL1);
        [OFL1_yfit, probability] = trainedModel.predictFcn(training_OFL1); 
        [~, ~, ~, train_AUC] = perfcurve(X_label_train_balanced, probability(:,2), 1);

        if sum(test_labels) == 0
            test_AUC = 0; 
        else
            if perm == 0
                Var = transpose([test_zscore; test_labels]);
                Var(:,cell_rem) = [];
                test_OFL2 = array2table(Var);
                [OFL2_yfit, probability] = trainedModel.predictFcn(test_OFL2);
                [~, ~, ~, test_AUC] = perfcurve(test_labels, probability(:,2), 1);
            else
                test_AUC = [];
                for i=1:perm
                    random_test_labels = test_labels(randperm(length(test_labels)));

                    Var = transpose([test_zscore; random_test_labels]);
                    Var(:,cell_rem) = [];
                    test_OFL2 = array2table(Var);
                    [OFL2_yfit, probability] = trainedModel.predictFcn(test_OFL2);
                    [~, ~, ~, test_AUC_this] = perfcurve(random_test_labels, probability(:,2), 1);
                    test_AUC(length(test_AUC) + 1) = test_AUC_this;
                end
            end
        end
    else
        train_AUC = 0;
        test_AUC = 0; 
    end
end