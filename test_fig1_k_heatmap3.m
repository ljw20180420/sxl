% Depicting figures included in Fig1K

% load ['total_heatmap_signal.mat'] in Example_data folder for
% demonstration of Fig1K


%-----------------
% Date: 2024-04-30
% Author: Choi et al.
%-----------------



clear all


load('for_LJW/F2_3/df_f_zscore.npy.mat')
auroc_table = readtable("for_LJW/auroc_analysis.csv");
mouse = 'F2_3';
event = 'label1'; % label1, label2, label3, label4, label5
event_idx = 1:6299;
blank_frame = 2321;
strong_frame = 2343;
types = ["excited", "inhibited", "non-responsive"]
select_num = 50;

figure(1)
tiledlayout(6,1);
for tt=1:length(types)
    type = types(tt);
    % select mouse and event and type
    type_cells = auroc_table(string(auroc_table.mouse) == mouse & string(auroc_table.label) == event & string(auroc_table.type) == type, 'cell');
    type_idx = sort(cellfun(@str2num, regexprep(type_cells.cell, '^cell', ''))) + 1;

    df_f_zscore_event_type = df_f_zscore(type_idx, event_idx);
    df_f_zscore_dec{tt} = zscore_sorted(df_f_zscore_event_type, 'descend', 2238, 2259, 10.38);

    ax = nexttile
    clims=[0, 10];
    % if type == 'inhibited'
    %     clims=[-5, 0];
    % else
    %     clims=[0, 10];
    % end
    imagesc(df_f_zscore_dec{tt}(1:min(select_num, end), :), clims)
    % imagesc(df_f_zscore_dec{tt}(1:min(select_num, end), :))
    set(gca,'TickDir','out')
    if type == 'inhibited'
        load('cold.mat')
        % colormap(ax, cold)
        colormap(ax, flipud(cold))
    else
        colormap(ax, "hot")
    end
    colorbar
    nexttile
    plot(mean(df_f_zscore_dec{tt}))
    xlim([0, size(df_f_zscore_dec{tt}, 2)])
end






%%% representative figures for 50 neurons

% figure(1)
% tiledlayout(1,3);
% colormap hot

% nexttile
% clims=[0 10];
% imagesc(sorted_norm_OBfz_inc(1:50,:),clims)
% colorbar

% nexttile
% clims=[0 10];
% imagesc(sorted_norm_DMjump_inc(1:50,:),clims)
% colorbar

% nexttile
% clims=[0 10];
% imagesc(sorted_norm_DMfz_inc(1:50,:),clims)
% colorbar

% figure(2)
% tiledlayout(1,3);
% load('cold.mat')
% colormap(cold)
% nexttile
% clims=[-5 0];
% imagesc(sorted_norm_OBfz_dec(1:50,:),clims)
% colorbar

% nexttile
% clims=[-5 0];
% imagesc(sorted_norm_DMjump_dec(1:50,:),clims)
% colorbar

% nexttile
% clims=[-5 0];
% imagesc(sorted_norm_DMfz_dec(1:50,:),clims)
% colorbar


%%%%%%

% figure(3)
% tiledlayout(1,3);
% nexttile
% avg_OBfz_inc = mean(sorted_norm_OBfz_inc);
% plot(avg_OBfz_inc)
% xlim([0 50])
% ylim([-1 2.5])
% nexttile
% avg_DMjump_inc = mean(sorted_norm_DMjump_inc);
% plot(avg_DMjump_inc)
% xlim([0 50])
% ylim([-1 2.5])
% nexttile
% avg_DMfz_inc = mean(sorted_norm_DMfz_inc);
% plot(avg_DMfz_inc)
% xlim([0 50])
% ylim([-1 2.5])

% figure(4)
% tiledlayout(1,3);

% nexttile
% avg_OBfz_dec = mean(sorted_norm_OBfz_dec);
% plot(avg_OBfz_dec)
% xlim([0 50])
% ylim([-1.5 1])
% nexttile
% avg_DMjump_dec = mean(sorted_norm_DMjump_dec);
% plot(avg_DMjump_dec)
% xlim([0 50])
% ylim([-1.5 1])
% nexttile
% avg_DMfz_dec = mean(sorted_norm_DMfz_dec(1:86,:));
% plot(avg_DMfz_dec)
% xlim([0 50])
% ylim([-1.5 1])




function sorted_zscore = zscore_sorted(data, direction, blank_frame, strong_frame, frame_per_second)
    % 1 ~ blank_frame是没有信号的范围
    % blank_frame + 1 ~ strong_frame用来排序
    second_num = floor(length(data) / frame_per_second);
    cell_num = size(data,1);
    binned = zeros(cell_num, second_num);
    for i = 1:second_num
        start_frame = frame_per_second * (i - 1);
        start_weight = ceil(start_frame) - start_frame;
        start_frame = ceil(start_frame);
        end_frame = frame_per_second * i;
        end_weight = end_frame - floor(end_frame);
        end_frame = floor(end_frame) + 1;
        if start_frame > 0
            binned(:, i) = binned(:, i) + data(:, start_frame) * start_weight;
        end
        binned(:, i) = binned(:, i) + sum(data(:, start_frame + 1: end_frame - 1), 2);
        if end_frame <= length(data)
            binned(:, i) = binned(:, i) + data(:, end_frame) * end_weight;
        end
    end
    
    blanck_second = floor(blank_frame / frame_per_second);
    differ_mu = mean(binned(:, 1:blanck_second), 2);
    differ_std = std(binned(:, 1:blanck_second), 0, 2);
    zscore = (binned - differ_mu) ./ differ_std;
    
    strong_second = ceil(strong_frame / frame_per_second);
    differ_post = mean(zscore(:, (blanck_second + 1):strong_second), 2)
    [~, ind] = sort(differ_post, direction);
    sorted_zscore = zscore(ind,:);
end
