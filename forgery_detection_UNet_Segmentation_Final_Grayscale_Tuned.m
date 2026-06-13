%% forgery_detection_UNet_Segmentation_Final_Grayscale_Tuned_fixed.m
% Author: Adapted for Dr. Abhishek Thakur (final fix)
% Notes:
%  - U-Net segmentation only (forged region detection)
%  - On-the-fly mask processing (no processed masks saved)
%  - Save 5 Binary + 5 Overlay samples only
%  - Pixel-level Precision, Recall, F1, IoU, Confusion matrix (grayscale)
%  - Robust plotting and property usage compatible with MATLAB

clear; close all; clc;
warning('off','all');

%% ---------------- USER CONFIG ----------------
dataFolder    = 'D:\claude\SemanticSegmentationUsingFCN-AlexNet1\Dataset4\Images';
labelFolder   = 'D:\claude\SemanticSegmentationUsingFCN-AlexNet1\Dataset4\Labels';
resultsFolder = 'D:\claude\SemanticSegmentationUsingFCN-AlexNet1\Final_Segmentation_Results_Tuned';

if ~exist(resultsFolder,'dir'), mkdir(resultsFolder); end
binaryOutDir  = fullfile(resultsFolder,'BinarySamples');  if ~exist(binaryOutDir,'dir'), mkdir(binaryOutDir); end
overlayOutDir = fullfile(resultsFolder,'OverlaySamples'); if ~exist(overlayOutDir,'dir'), mkdir(overlayOutDir); end
figDir        = fullfile(resultsFolder,'Figures');        if ~exist(figDir,'dir'), mkdir(figDir); end

rng(42);

%% ---------------- PAIR IMAGE & MASK FILES ----------------
[imgList, maskList] = pair_images_and_masks(dataFolder, labelFolder);
if isempty(imgList)
    error('No matched image-mask pairs found. Check folders and basenames.');
end
fprintf('Found %d paired images.\n', numel(imgList));

%% ---------------- SPLIT TRAIN / VAL ----------------
fracTrain = 0.8;
Nall = numel(imgList);
idx = randperm(Nall);
nTrain = round(fracTrain * Nall);
if nTrain < 1, nTrain = 1; end
trainIdx = idx(1:nTrain);
valIdx   = idx(nTrain+1:end);
trainImgs = imgList(trainIdx); trainMasks = maskList(trainIdx);
valImgs   = imgList(valIdx);   valMasks   = maskList(valIdx);
fprintf('Train: %d | Validation: %d\n', numel(trainImgs), numel(valImgs));
if isempty(valImgs)
    error('Validation set is empty. Reduce training fraction or add images.');
end

%% ---------------- U-NET CONFIG ----------------
% EncoderDepth=4 => image dims must be multiples of 16
segInputSize = [352 480 3];    % [H W C] (352 and 480 are multiples of 16)
numClasses = 2;
classNames = ["Background","Forged"];
labelIDs = [1 2];  % 1 background, 2 forged

% Image datastores (they will be resized by ReadFcn)
imdsTrain = imageDatastore(trainImgs);
imdsVal   = imageDatastore(valImgs);
imdsTrain.ReadFcn = @(x) im2uint8(imresize(imread(x), segInputSize(1:2)));
imdsVal.ReadFcn   = @(x) im2uint8(imresize(imread(x), segInputSize(1:2)));

% Pixel label datastores (ReadFcn converts raw mask to label IDs 1/2 on-the-fly)
pxdsTrain = pixelLabelDatastore(trainMasks, classNames, labelIDs);
pxdsVal   = pixelLabelDatastore(valMasks,   classNames, labelIDs);
pxdsTrain.ReadFcn = @(x) maskReadFcn_noSave(x, segInputSize(1:2));
pxdsVal.ReadFcn   = @(x) maskReadFcn_noSave(x, segInputSize(1:2));

% Combine for training
dsTrain = pixelLabelImageDatastore(imdsTrain, pxdsTrain);
dsVal   = pixelLabelImageDatastore(imdsVal, pxdsVal);

%% ---------------- NETWORK & TRAINING OPTIONS ----------------
encoderDepth = 4;
lgraph = unetLayers(segInputSize, numClasses, 'EncoderDepth', encoderDepth);

opts = trainingOptions('adam', ...
    'InitialLearnRate',5e-4, ...
    'MaxEpochs',12, ...
    'MiniBatchSize',8, ...
    'Shuffle','every-epoch', ...
    'ValidationData', dsVal, ...
    'ValidationFrequency', max(1,ceil(numel(trainImgs)/8)), ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'ExecutionEnvironment','auto');

fprintf('Training U-Net (this may take time)...\n');
[netSeg, info] = trainNetwork(dsTrain, lgraph, opts);
save(fullfile(resultsFolder,'netSeg_final.mat'),'netSeg','info','-v7.3');
fprintf('Training finished and saved.\n');

%% ---------------- PREDICT & SAVE 5 SAMPLES ----------------
numVal = numel(valImgs);
nSave = min(5, numVal);
savedCount = 0;

yTrue_all = [];
yPred_all = [];

for i = 1:numVal
    % read full-res image and mask
    I_full = imread(valImgs{i});
    M_full = imread(valMasks{i});
    if size(M_full,3)>1, M_full = rgb2gray(M_full); end

    % predict (resize for network then upsample)
    I_res = im2uint8(imresize(I_full, segInputSize(1:2)));
    predCat = semanticseg(I_res, netSeg);
    predForged_res = (predCat == "Forged");
    predForged = imresize(predForged_res, [size(I_full,1), size(I_full,2)], 'nearest');
    predLabelMap = uint8(predForged) + uint8(1); % 1 background, 2 forged

    % ground truth label map (1/2)
    gtBinary = M_full > 127;
    gtLabelMap = uint8(gtBinary) + uint8(1);

    % ensure same dims
    if ~isequal(size(gtLabelMap), size(predLabelMap))
        predLabelMap = imresize(predLabelMap, size(gtLabelMap), 'nearest');
        predForged = predLabelMap == 2;
    end

    % accumulate for metrics
    yTrue_all = [yTrue_all; double(gtLabelMap(:))]; %#ok<AGROW>
    yPred_all = [yPred_all; double(predLabelMap(:))]; %#ok<AGROW>

    % save sample binary + overlay (only up to nSave)
    if savedCount < nSave
        savedCount = savedCount + 1;
        [~, base, ~] = fileparts(valImgs{i});

        % binary: white background 255, forged black 0
        binImg = uint8(255 * ones(size(predForged),'uint8'));
        binImg(predForged) = 0;
        imwrite(binImg, fullfile(binaryOutDir, sprintf('Binary_%s.png', base)));

        % overlay: black out forged region on original (preserve other pixels)
        if size(I_full,3) == 1, I_vis = cat(3,I_full,I_full,I_full); else I_vis = I_full; end
        overlay = I_vis;
        mask3 = repmat(predForged, [1 1 3]);
        overlay(mask3) = 0;
        imwrite(overlay, fullfile(overlayOutDir, sprintf('Overlay_%s.png', base)));
    end
end

fprintf('Saved %d sample Binary & Overlay images.\n', savedCount);

%% ---------------- METRICS ----------------
classesOrder = [1 2]; % background then forged
cm = confusionmat(yTrue_all, yPred_all, 'Order', classesOrder);

TP = diag(cm);
FP = sum(cm,1)' - TP;
FN = sum(cm,2) - TP;

precision = TP ./ (TP + FP + eps);
recall    = TP ./ (TP + FN + eps);
f1score   = 2 .* (precision .* recall) ./ (precision + recall + eps);
IoU = TP ./ (sum(cm,2) + sum(cm,1)' - TP + eps);
globalAcc = sum(diag(cm)) / sum(cm(:));

T = table(["Background";"Forged"], precision, recall, f1score, IoU, ...
    'VariableNames', {'Class','Precision','Recall','F1','IoU'});
disp('Per-class pixel metrics:'); disp(T);
fprintf('Global pixel accuracy: %.4f\n', globalAcc);

writetable(T, fullfile(resultsFolder,'PerClass_PixelMetrics.csv'));
save(fullfile(resultsFolder,'PixelMetrics_final.mat'),'cm','precision','recall','f1score','IoU','globalAcc','T');

%% ---------------- PLOTTING (GRAYSCALE) ----------------
% Confusion matrix (counts)
fig1 = figure('Color','w','Visible','off');
imagesc(cm); axis equal tight;
colormap(gca, gray);
colorbar;
xticks(1:2); yticks(1:2); xticklabels({'Background','Forged'}); yticklabels({'Background','Forged'});
xlabel('Predicted'); ylabel('True'); title('Pixel-level Confusion Matrix (counts)');
for r=1:2
    for c=1:2
        val = cm(r,c);
        text(c, r, sprintf('%d', val), 'HorizontalAlignment','center', 'Color','w', 'FontSize', 11);
    end
end
saveas(fig1, fullfile(figDir,'ConfusionMatrix_counts.png'));
close(fig1);

% Precision/Recall/F1 (grayscale dashed/varied)
fig2 = figure('Color','w','Visible','off'); hold on; grid on;
x = 1:2;
plot(x, precision, '--o', 'LineWidth',1.6, 'Color',[0 0 0]);
plot(x, recall,    ':s', 'LineWidth',1.6, 'Color',[0.4 0.4 0.4]);
plot(x, f1score,   '-.^', 'LineWidth',1.6, 'Color',[0.6 0.6 0.6]);
xticks(x); xticklabels({'Background','Forged'}); xtickangle(0);
xlabel('Class'); ylabel('Metric'); title('Pixel-level Precision / Recall / F1');
legend({'Precision','Recall','F1'}, 'Location','best'); colormap(gca, gray);
saveas(fig2, fullfile(figDir,'Precision_Recall_F1.png'));
close(fig2);

% Training Loss & (if present) Accuracy
fig3 = figure('Color','w','Visible','off');
subplot(2,1,1);
hold on; grid on;
if isfield(info,'TrainingLoss') && ~isempty(info.TrainingLoss)
    plot(info.TrainingLoss, '--', 'LineWidth',1.4, 'Color',[0 0 0]);
end
if isfield(info,'ValidationLoss') && ~isempty(info.ValidationLoss)
    plot(info.ValidationLoss, ':', 'LineWidth',1.4, 'Color',[0.4 0.4 0.4]);
end
title('Training & Validation Loss'); legend('Train','Val'); xlabel('Iteration'); ylabel('Loss');

subplot(2,1,2);
plotted = false;
hold on; grid on;
if isfield(info,'TrainingAccuracy') && ~isempty(info.TrainingAccuracy)
    plot(info.TrainingAccuracy, '--', 'LineWidth',1.4, 'Color',[0 0 0]); plotted = true;
end
if isfield(info,'ValidationAccuracy') && ~isempty(info.ValidationAccuracy)
    plot(info.ValidationAccuracy, ':', 'LineWidth',1.4, 'Color',[0.4 0.4 0.4]); plotted = true;
end
if plotted
    title('Training & Validation Accuracy'); legend('Train','Val'); xlabel('Iteration'); ylabel('Accuracy');
else
    title('No training/validation accuracy available'); axis off;
end
colormap(gca, gray);
saveas(fig3, fullfile(figDir,'TrainVal_LossAcc.png'));
close(fig3);

fprintf('\nAll outputs saved to: %s\n', resultsFolder);
fprintf('  - Binary samples: %s\n  - Overlay samples: %s\n  - Figures: %s\n', binaryOutDir, overlayOutDir, figDir);

%% ---------------- LOCAL FUNCTIONS ----------------

function out = maskReadFcn_noSave(maskFile, targetSize)
    % Read mask, convert to single channel, binarize and convert to labels 1/2, resize nearest.
    M = imread(maskFile);
    if size(M,3) > 1, M = rgb2gray(M); end
    if islogical(M)
        bw = M;
    else
        % Otsu is robust; fallback to threshold 127 if it fails
        try
            lvl = graythresh(M);
            bw = imbinarize(M, lvl);
        catch
            bw = M > 127;
        end
    end
    lab = uint8(bw) + uint8(1); % background=1, forged=2
    out = imresize(lab, targetSize, 'nearest');
end

function [imgList, maskList] = pair_images_and_masks(imgDir, maskDir)
    imgFiles = listFilesWithExt(imgDir, {'.jpg','.jpeg','.png','.tif','.tiff'});
    maskFiles = listFilesWithExt(maskDir, {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'});
    imgBases = cellfun(@(p)lower(stripExtension(p)), imgFiles, 'UniformOutput', false);
    maskBases = cellfun(@(p)lower(stripExtension(p)), maskFiles, 'UniformOutput', false);
    imgList = {}; maskList = {};
    for ii = 1:numel(imgFiles)
        k = find(strcmp(imgBases{ii}, maskBases), 1);
        if ~isempty(k)
            imgList{end+1,1} = imgFiles{ii}; %#ok<AGROW>
            maskList{end+1,1} = maskFiles{k}; %#ok<AGROW>
        end
    end
end

function out = listFilesWithExt(folder, exts)
    out = {};
    for e = exts
        files = dir(fullfile(folder, ['*' e{1}]));
        for k = 1:numel(files)
            if ~files(k).isdir
                out{end+1,1} = fullfile(folder, files(k).name); %#ok<AGROW>
            end
        end
    end
end

function s = stripExtension(p)
    [~, n, ~] = fileparts(p); s = n;
end
