%% plot_and_compare_all_models_fixed.m
% Load saved model/results for classification & segmentation, compute metrics,
% and produce comparison visualizations (per-class line graphs, macro bar charts,
% confusion matrices, and ROC curves).
%
% Edit 'resultsFolder' to point to your results directory and run the file.

clearvars; close all; clc;

%% User config: point this to your results folder
resultsFolder = 'D:\claude\SemanticSegmentationUsingFCN-AlexNet1\forgery_results_v2';
figOut = fullfile(resultsFolder,'comparison_figs');
if ~exist(figOut,'dir'), mkdir(figOut); end

%% Expected result filenames (change if different)
resFiles.ResNet50 = fullfile(resultsFolder,'ResNet50_class.mat');        % may contain YPred/YTrue or netRes50
resFiles.MobileNet = fullfile(resultsFolder,'MobileNetv2_class.mat');

resFiles.FCN_AlexNet = fullfile(resultsFolder,'FCN_AlexNet_metrics.mat');    % expects variable metrics
resFiles.UNet_ResNet18 = fullfile(resultsFolder,'UNet_ResNet18_metrics.mat');
resFiles.DeepLabv3p = fullfile(resultsFolder,'DeepLabv3p_metrics.mat');

%% Models lists
classificationModels = {'ResNet50','MobileNet'};
segModels = {'FCN_AlexNet','UNet_ResNet18','DeepLabv3p'};

%% Load classification saved results or networks
classResults = struct();
for i = 1:numel(classificationModels)
    m = classificationModels{i};
    fname = resFiles.(m);
    if exist(fname,'file')
        S = load(fname);
        % If saved predictions exist, load them
        if isfield(S,'YPred') && isfield(S,'YTrue')
            classResults.(m).YPred = categorical(S.YPred);
            classResults.(m).YTrue = categorical(S.YTrue);
            if isfield(S,'scores'), classResults.(m).scores = S.scores; end
            fprintf('Loaded YPred/YTrue for %s from %s\n', m, fname);
        else
            % Try to find a network object
            if isfield(S,'netRes50'), classResults.(m).net = S.netRes50; fprintf('Loaded netRes50 for %s\n', m); end
            if isfield(S,'netMobile'), classResults.(m).net = S.netMobile; fprintf('Loaded netMobile for %s\n', m); end
            if isfield(S,'net') && ~isfield(classResults.(m),'net'), classResults.(m).net = S.net; fprintf('Loaded net for %s\n', m); end
            if ~isfield(classResults.(m),'net')
                warning('File %s does not contain YPred/YTrue or network for model %s.', fname, m);
            else
                classResults.(m).needsPredict = true;
            end
        end
    else
        warning('Results file not found for model %s: %s', m, fname);
    end
end

%% Compute classification metrics where predictions exist
classSummary = struct();
for i = 1:numel(classificationModels)
    m = classificationModels{i};
    if isfield(classResults.(m),'YPred') && isfield(classResults.(m),'YTrue')
        YP = categorical(classResults.(m).YPred);
        YT = categorical(classResults.(m).YTrue);
        classes = categories(YT);
        C = confusionmat(YT,YP,'Order',classes);
        TP = diag(C);
        FP = sum(C,1)' - TP;
        FN = sum(C,2) - TP;
        precision = TP ./ (TP + FP + eps);
        recall = TP ./ (TP + FN + eps);
        f1 = 2 .* (precision .* recall) ./ (precision + recall + eps);
        classSummary.(m).classes = classes;
        classSummary.(m).confMat = C;
        classSummary.(m).precision = precision;
        classSummary.(m).recall = recall;
        classSummary.(m).f1 = f1;
        classSummary.(m).macroPrecision = mean(precision);
        classSummary.(m).macroRecall = mean(recall);
        classSummary.(m).macroF1 = mean(f1);
        classSummary.(m).YPred = YP;
        classSummary.(m).YTrue = YT;
        fprintf('Computed classification metrics for %s (macroF1=%.4f)\n', m, classSummary.(m).macroF1);
    else
        fprintf('Skipping classification metrics for %s (no YPred/YTrue available).\n', m);
    end
end

%% Load segmentation metrics (if saved)
segSummary = struct();
for i = 1:numel(segModels)
    m = segModels{i};
    fname = resFiles.(m);
    if exist(fname,'file')
        S = load(fname);
        if isfield(S,'metrics')
            metrics = S.metrics;
            try
                cm = metrics.ClassMetrics;
                tblVars = cm.Properties.VariableNames;
                pcol = intersect(tblVars, {'Precision','precision'});
                rcol = intersect(tblVars, {'Recall','recall'});
                fcol = intersect(tblVars, {'F1Score','F1','f1'});
                if isempty(pcol) || isempty(rcol) || isempty(fcol)
                    % fallback use IoU
                    iouCol = intersect(tblVars, {'IoU','iou','MeanIoU'});
                    if ~isempty(iouCol)
                        iou = cm{:, iouCol{1}};
                        segSummary.(m).classes = cm.Row;
                        segSummary.(m).precision = iou;
                        segSummary.(m).recall = iou;
                        segSummary.(m).f1 = iou;
                    else
                        warning('Unable to extract P/R/F or IoU from metrics for %s', m);
                    end
                else
                    segSummary.(m).classes = cm.Row;
                    segSummary.(m).precision = cm{:, pcol{1}};
                    segSummary.(m).recall = cm{:, rcol{1}};
                    segSummary.(m).f1 = cm{:, fcol{1}};
                end
                segSummary.(m).metrics = metrics;
                fprintf('Loaded segmentation metrics for %s\n', m);
            catch ME
                warning('Error parsing metrics for %s: %s', m, ME.message);
            end
        else
            warning('Metrics variable not found in %s', fname);
        end
    else
        warning('Segmentation metrics file not found: %s', fname);
    end
end

%% Plotting: per-class line plots for classification (if available)
classModelsAvailable = {};
for i=1:numel(classificationModels)
    m = classificationModels{i};
    if isfield(classSummary,m), classModelsAvailable{end+1} = m; end
end
if ~isempty(classModelsAvailable)
    plot_per_class(classModelsAvailable, classSummary, figOut, 'Classification');
end

%% Plotting: per-class line plots for segmentation (if available)
segModelsAvailable = {};
for i=1:numel(segModels)
    m = segModels{i};
    if isfield(segSummary,m), segModelsAvailable{end+1} = m; end
end
if ~isempty(segModelsAvailable)
    plot_per_class(segModelsAvailable, segSummary, figOut, 'Segmentation');
end

%% Macro-average bar charts - classification
if ~isempty(classModelsAvailable)
    models = classModelsAvailable;
    P = zeros(1,numel(models)); R = zeros(1,numel(models)); F = zeros(1,numel(models));
    labs = models;
    for i=1:numel(models)
        m = models{i};
        P(i) = classSummary.(m).macroPrecision;
        R(i) = classSummary.(m).macroRecall;
        F(i) = classSummary.(m).macroF1;
    end
    h = figure('Visible','on'); bar([P;R;F]'); set(gca,'XTickLabel',labs); legend({'Precision','Recall','F1'}); ylabel('Macro-average'); title('Classification Macro metrics'); saveas(h, fullfile(figOut,'classification_macro_bar.png')); close(h);
end

%% Macro-average bar charts - segmentation
if ~isempty(segModelsAvailable)
    models = segModelsAvailable;
    P = zeros(1,numel(models)); R = zeros(1,numel(models)); F = zeros(1,numel(models));
    labs = models;
    for i=1:numel(models)
        m = models{i};
        P(i) = mean(segSummary.(m).precision);
        R(i) = mean(segSummary.(m).recall);
        F(i) = mean(segSummary.(m).f1);
    end
    h = figure('Visible','on'); bar([P;R;F]'); set(gca,'XTickLabel',labs); legend({'Precision','Recall','F1'}); ylabel('Macro-average'); title('Segmentation Macro metrics'); saveas(h, fullfile(figOut,'segmentation_macro_bar.png')); close(h);
end

%% Confusion matrices for classification models
for i=1:numel(classModelsAvailable)
    m = classModelsAvailable{i};
    YT = classSummary.(m).YTrue;
    YP = classSummary.(m).YPred;
    h = figure('Visible','on'); cm = confusionchart(YT, YP); cm.Title = ['Confusion matrix - ' m]; saveas(h, fullfile(figOut,['confusion_' m '.png'])); close(h);
end

%% ROC curves (one-vs-rest) for classification models if scores available
colors = lines(8);
for i=1:numel(classModelsAvailable)
    m = classModelsAvailable{i};
    if isfield(classResults.(m),'scores')
        scores = classResults.(m).scores;
        YT = classSummary.(m).YTrue;
        classes = categories(YT);
        numC = numel(classes);
        figure('Visible','on'); hold on;
        legendEntries = cell(1,numC);
        for k=1:numC
            ybin = double(YT == classes{k});
            probs = scores(:,k);
            [X,Y,T,AUC] = perfcurve(ybin, probs, 1);
            plot(X,Y,'LineWidth',1.6,'Color',colors(k,:));
            legendEntries{k} = sprintf('%s (AUC=%.3f)', string(classes{k}), AUC);
        end
        xlabel('False positive rate'); ylabel('True positive rate'); title(['ROC (one-vs-rest) - ' m]);
        legend(legendEntries,'Location','best'); grid on;
        saveas(gcf, fullfile(figOut,['ROC_' m '.png'])); close(gcf);
    else
        fprintf('Skipping ROC for %s: no scores available.\n', m);
    end
end

fprintf('All plots (that could be generated) saved to: %s\n', figOut);

%% ------------------------------------------------------------------------
%% Local helper functions (all functions must be at file end for scripts)
%% ------------------------------------------------------------------------

function plot_per_class(modelsList, summaryStruct, outFolder, outPrefix)
    modelColors = lines(6);
    lineStyles = {'-','--',':','-.','-','--'};
    markerStyles = {'o','s','^','d','v','p'};
    % Determine reference classes from first available model
    refClasses = [];
    for ii=1:numel(modelsList)
        m = modelsList{ii};
        if isfield(summaryStruct,m)
            refClasses = summaryStruct.(m).classes;
            break;
        end
    end
    if isempty(refClasses)
        warning('No class info available for %s plots.', outPrefix);
        return;
    end
    K = numel(refClasses);
    x = 1:K;
    legendEntries = {};
    % Precision
    h = figure('Name',[outPrefix '_precision_per_class'],'Visible','on','Position',[200 200 1000 420]); hold on;
    for ii=1:numel(modelsList)
        m = modelsList{ii};
        if isfield(summaryStruct,m)
            p = summaryStruct.(m).precision;
            plot(x, p, 'LineStyle', lineStyles{mod(ii-1,length(lineStyles))+1}, 'Marker', markerStyles{mod(ii-1,length(markerStyles))+1}, 'LineWidth',1.8, 'Color', modelColors(ii,:));
            legendEntries{end+1} = m; %#ok<AGROW>
        end
    end
    xlim([1 K]); xticks(x); xticklabels(refClasses); xtickangle(45); ylabel('Precision'); xlabel('Class'); grid on; title([outPrefix ' - Precision per class']);
    legend(legendEntries,'Interpreter','none','Location','best'); saveas(h, fullfile(outFolder,[outPrefix '_precision_per_class.png'])); close(h);
    % Recall
    h = figure('Name',[outPrefix '_recall_per_class'],'Visible','on','Position',[200 200 1000 420]); hold on;
    for ii=1:numel(modelsList)
        m = modelsList{ii};
        if isfield(summaryStruct,m)
            r = summaryStruct.(m).recall;
            plot(x, r, 'LineStyle', lineStyles{mod(ii-1,length(lineStyles))+1}, 'Marker', markerStyles{mod(ii-1,length(markerStyles))+1}, 'LineWidth',1.8, 'Color', modelColors(ii,:));
        end
    end
    xlim([1 K]); xticks(x); xticklabels(refClasses); xtickangle(45); ylabel('Recall'); xlabel('Class'); grid on; title([outPrefix ' - Recall per class']);
    legend(legendEntries,'Interpreter','none','Location','best'); saveas(h, fullfile(outFolder,[outPrefix '_recall_per_class.png'])); close(h);
    % F1
    h = figure('Name',[outPrefix '_f1_per_class'],'Visible','on','Position',[200 200 1000 420]); hold on;
    for ii=1:numel(modelsList)
        m = modelsList{ii};
        if isfield(summaryStruct,m)
            f = summaryStruct.(m).f1;
            plot(x, f, 'LineStyle', lineStyles{mod(ii-1,length(lineStyles))+1}, 'Marker', markerStyles{mod(ii-1,length(markerStyles))+1}, 'LineWidth',1.8, 'Color', modelColors(ii,:));
        end
    end
    xlim([1 K]); xticks(x); xticklabels(refClasses); xtickangle(45); ylabel('F1'); xlabel('Class'); grid on; title([outPrefix ' - F1 per class']);
    legend(legendEntries,'Interpreter','none','Location','best'); saveas(h, fullfile(outFolder,[outPrefix '_f1_per_class.png'])); close(h);
end

function v = getfieldifexists(s, name)
    if isfield(s,name), v = s.(name); else v = NaN; end
end
