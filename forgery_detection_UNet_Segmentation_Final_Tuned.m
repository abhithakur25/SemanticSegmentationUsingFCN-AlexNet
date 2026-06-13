%% forgery_detection_UNet_Segmentation_Final_Tuned.m
% Author: Adapted for Dr. Abhishek Thakur (ChatGPT)
% Purpose:
%   - Train U-Net segmentation to detect forged patches
%   - Do NOT save any processed masks to disk (masks processed on-the-fly)
%   - Save only 5 BinaryMasks and 5 Overlay images (forged=black, background=white)
%   - Compute pixel-level Precision, Recall, F1, IoU, Confusion Matrix and plots (grayscale)
% Notes:
%   - Requires Deep Learning Toolbox and Image Processing Toolbox
%   - Ensure image & mask filenames have matching basenames for pairing

clear; close all; clc;
warning('off','all');

%% ---------------- USER CONFIG ----------------
dataFolder    = 'D:\claude\SemanticSegmentationUsingFCN-AlexNet1\Dataset4\Images';
labelFolder   = 'D:\claude\SemanticSegmentationUsingFCN-AlexNet1\Dataset4\Labels';
resultsFolder = 'D:\claude\SemanticSegmentationUsingFCN-AlexNet1\Final_Segmentation_Results_Adapted';

if ~exist(resultsFolder,'dir'), mkdir(resultsFolder); end
binaryOutDir = fullfile(resultsFolder,'BinarySamples'); if ~exist(binaryOutDir,'dir'), mkdir(binaryOutDir); end
overlayOutDir = fullfile(resultsFolder,'OverlaySamples'); if ~exist(overlayOutDir,'dir'), mkdir(overlayOutDir); end
figDir = fullfile(resultsFolder,'Figures'); if ~exist(figDir,'dir'), mkdir(figDir); end
csvFile = fullfile(resultsFolder,'PerClassPixelMetrics.csv');

rng(0);

%% ---------------- PAIR IMAGE & MASK FILES ----------------
[imgList, maskList] = pair_images_and_masks(dataFolder, labelFolder);
if isempty(imgList)
    error('No matched image-mask pairs found. Check folders and basenames.');
end
N = numel(imgList);
fprintf('Found %d paired images.\n', N);

%% ---------------- SPLIT TRAIN / VAL ----------------
fracTrain = 0.8;
idx = randperm(N);
nTrain = max(1, round(fracTrain * N));
trainIdx = idx(1:nTrain);
valIdx   = idx(nTrain+1:end);
if isempty(valIdx)
    error('Validation set is empty. Reduce training fraction or add images.');
end

trainImgs = imgList(trainIdx);
trainMasks = maskList(trainIdx);
valImgs = imgList(valIdx);
valMasks = maskList(valIdx);

fprintf('Train images: %d  |  Val images: %d\n', numel(trainImgs), numel(valImgs));

%% ---------------- SEGMENTATION INPUT SIZE & CLASSES ----------------
% For EncoderDepth=4, height and width must be multiples of 16 (2^4).
segInputSize = [352 480 3];  % [H W C] -> H divisible by 16 (352 OK), W divisible by 16 (480 OK)
numClasses = 2;
classNames = ["Background","Forged"];
labelIDs = [1 2];  % 1 -> background, 2 -> forged

%% ---------------- CREATE IMAGE / PIXELLABEL DATASTORES (ON-THE-FLY MASK PROCESSING) ----------------
% imageDatastores
imdsTrain = imageDatastore(trainImgs);
imdsVal   = imageDatastore(valImgs);

% pixelLabelDatastores: pass file lists, but set ReadFcn to produce numeric label map (1/2) without saving
pxdsTrain = pixelLabelDatastore(trainMasks, classNames, labelIDs);
pxdsVal   = pixelLabelDatastore(valMasks, classNames, labelIDs);

% Set ReadFcn to read, convert to grayscale, threshold and resize (nearest)
pxdsTrain.ReadFcn = @(x) maskReadFcn_noSave(x, segInputSize(1:2));
pxdsVal.ReadFcn   = @(x) maskReadFcn_noSave(x, segInputSize(1:2));

% Set image ReadFcn to resize to segInputSize
imdsTrain.ReadFcn = @(x) im2uint8(imresize(imread(x), segInputSize(1:2)));
imdsVal.ReadFcn   = @(x) im2uint8(imresize(imread(x), segInputSize(1:2)));

% Combine into pixelLabelImageDatastore for training/validation
dsTrain = pixelLabelImageDatastore(imdsTrain, pxdsTrain);
dsVal   = pixelLabelImageDatastore(imdsVal, pxdsVal);

%% ---------------- BUILD U-NET ----------------
encoderDepth = 4;
lgraph = unetLayers(segInputSize, numClasses, 'EncoderDepth', encoderDepth);

%% ---------------- TRAINING OPTIONS ----------------
opts = trainingOptions('adam', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',8, ...
    'MiniBatchSize',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData', dsVal, ...
    'ValidationFrequency', max(1,floor(numel(trainImgs)/4)), ...
    'Verbose',false, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment','auto');

%% ---------------- TRAIN NETWORK ----------------
fprintf('Training U-Net... (this may take time)\n');
[netSeg, info] = trainNetwork(dsTrain, lgraph, opts);
save(fullfile(resultsFolder,'netSeg_final_tuned.mat'),'netSeg','info','-v7.3');
fprintf('Training finished and model saved.\n');

%% ---------------- PREDICTION ON VALIDATION & SAVE 5 SAMPLES ----------------
numVal = numel(valImgs);
nSamplesToSave = min(5, numVal);
savedCount = 0;

yTrue_all = [];
yPred_all = [];

for i = 1:numVal
    % Read original full-resolution image and mask (not resized)
    I_full = imread(valImgs{i});
    M_full = imread(valMasks{i});
    if size(M_full,3) > 1, M_full = rgb2gray(M_full); end

    % Prepare input for network (resized)
    I_res = im2uint8(imresize(I_full, segInputSize(1:2)));
    predCat = semanticseg(I_res, netSeg);
    predForged_res = (predCat == "Forged");

    % Resize prediction back to original image size (nearest)
    predForged = imresize(predForged_res, [size(I_full,1) size(I_full,2)], 'nearest');
    predLabelMap = uint8(predForged) + uint8(1); % 0->1 background, 1->2 forged

    % Convert GT to label map (1/2) on-the-fly
    gtBinary = M_full > 127;
    gtLabelMap = uint8(gtBinary) + uint8(1);

    % Ensure same size
    if ~isequal(size(gtLabelMap), size(predLabelMap))
        predLabelMap = imresize(predLabelMap, size(gtLabelMap), 'nearest');
    end

    % accumulate for metrics
    yTrue_all = [yTrue_all; double(gtLabelMap(:))]; %#ok<AGROW>
    yPred_all = [yPred_all; double(predLabelMap(:))]; %#ok<AGROW>

    % Save up to nSamplesToSave examples (binary + overlay)
    if savedCount < nSamplesToSave
        savedCount = savedCount + 1;
        [~, base, ~] = fileparts(valImgs{i});

        % binary image (white background 255, forged black 0)
        binImg = uint8(255 * ones(size(predForged), 'uint8'));
        binImg(predForged) = 0;
        imwrite(binImg, fullfile(binaryOutDir, sprintf('Binary_%s.png', base)));

        % overlay: black-out the forged region on original image (preserve others)
        if size(I_full,3) == 1, I_vis = cat(3,I_full,I_full,I_full); else I_vis = I_full; end
        overlay = I_vis;
        mask3 = repmat(predForged, [1 1 3]);
        overlay(mask3) = 0;
        imwrite(overlay, fullfile(overlayOutDir, sprintf('Overlay_%s.png', base)));
    end
end

fprintf('Saved %d sample binary & overlay images to:\n%s\n%s\n', savedCount, binaryOutDir, overlayOutDir);

%% ---------------- COMPUTE PIXEL-LEVEL METRICS ----------------
classesOrder = [1 2]; % background then forged
cm = confusionmat(yTrue_all, yPred_all, 'Order', classesOrder);

TP = diag(cm);
FP = sum(cm,1)' - TP;
FN = sum(cm,2) - TP;

precision = TP ./ (TP + FP + eps);
recall = TP ./ (TP + FN + eps);
f1 = 2 * (precision .* recall) ./ (precision + recall + eps);
IoU = TP ./ (sum(cm,2) + sum(cm,1)' - TP + eps);
globalAcc = sum(diag(cm))/sum(cm(:));

% Save numeric metrics CSV
T = table(["Background";"Forged"], precision, recall, f1, IoU, 'VariableNames', {'Class','Precision','Recall','F1','IoU'});
writetable(T, csvFile);

% Save MATLAB mat
save(fullfile(resultsFolder,'PixelMetrics_final.mat'),'cm','precision','recall','f1','IoU','globalAcc','T');

%% ---------------- PLOT & SAVE FIGURES (grayscale) ----------------
% Confusion matrix: display counts and normalized %
fig1 = figure('Color','w');
imagesc(cm); axis equal tight; colormap(gray); colorbar;
title('Pixel-level Confusion Matrix (counts)');
xlabel('Predicted'); ylabel('True');
xticks(1:2); yticks(1:2); xticklabels({'Background','Forged'}); yticklabels({'Background','Forged'});
maxv = max(cm(:));
for r=1:2
    for c=1:2
        val = cm(r,c);
        percent = 100 * val / (sum(cm(:)) + eps);
        txt = sprintf('%d\n(%.2f%%)', val, percent);
        text(c, r, txt, 'HorizontalAlignment','center', 'Color', 'w', 'FontSize', 10);
    end
end
saveas(fig1, fullfile(figDir,'ConfusionMatrix_counts_pct.png'));

% Precision / Recall / F1 plot
fig2 = figure('Color','w'); hold on; grid on;
x = 1:2;
plot(x, precision, 'k--o','LineWidth',1.6);
plot(x, recall,    'k:*','LineWidth',1.6);
plot(x, f1,        'k-.+','LineWidth',1.6);
xticks(x); xticklabels({'Background','Forged'});
xlabel('Class'); ylabel('Metric'); title('Pixel-level Precision / Recall / F1');
legend({'Precision (---)','Recall (***)','F1 (-*-)'}, 'Location','best');
saveas(fig2, fullfile(figDir,'Precision_Recall_F1.png'));

% Save training loss & (if available) accuracy
fig3 = figure('Color','w');
subplot(2,1,1);
if isfield(info,'TrainingLoss')
    plot(info.TrainingLoss,'k','LineWidth',1.4); hold on;
end
if isfield(info,'ValidationLoss')
    plot(info.ValidationLoss,'k--','LineWidth',1.4);
end
title('Training & Validation Loss'); legend('Training','Validation');
xlabel('Iteration'); ylabel('Loss'); grid on;
subplot(2,1,2);
% info may not contain accuracy fields for segmentation; check and plot if available
plotted = false;
if isfield(info,'TrainingAccuracy'); plot(info.TrainingAccuracy,'k','LineWidth',1.4); hold on; plotted=true; end
if isfield(info,'ValidationAccuracy'); plot(info.ValidationAccuracy,'k--','LineWidth',1.4); plotted=true; end
if plotted
    title('Training & Validation Accuracy'); xlabel('Iteration'); ylabel('Accuracy'); legend('Train','Val'); grid on;
else
    title('No training/validation accuracy available in info'); axis off;
end
saveas(fig3, fullfile(figDir,'Train_Loss_Accuracy.png'));

fprintf('\n=== Summary ===\n');
disp(T);
fprintf('Global Pixel Accuracy: %.4f\n', globalAcc);
fprintf('All outputs (metrics, figures and 5 samples) saved to: %s\n', resultsFolder);

%% ---------------- LOCAL FUNCTIONS ----------------
function out = maskReadFcn_noSave(maskFile, targetSize)
    % Read a mask file, convert to single-channel, threshold, and resize with 'nearest'
    M = imread(maskFile);
    if size(M,3) > 1
        M = rgb2gray(M);
    end
    % If mask already logical, coerce to numeric
    if islogical(M)
        bw = M;
    else
        % use robust thresholding with Otsu; but if mask contains only 0/1 or 0/255 values, threshold is fine
        try
            level = graythresh(M);
            bw = imbinarize(M, level);
        catch
            bw = M > 127;
        end
    end
    % Map to label ids: background=1, forged=2
    lab = uint8(bw) + uint8(1);
    out = imresize(lab, targetSize, 'nearest');
end

function [imgList, maskList] = pair_images_and_masks(imgDir, maskDir)
    imgListAll = listFilesWithExt(imgDir, {'.jpg','.jpeg','.png','.tif','.tiff'});
    maskListAll = listFilesWithExt(maskDir, {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'});
    imgBases = cellfun(@(p)lower(stripExtension(p)), imgListAll, 'UniformOutput', false);
    maskBases = cellfun(@(p)lower(stripExtension(p)), maskListAll, 'UniformOutput', false);
    imgList = {}; maskList = {};
    for i=1:numel(imgListAll)
        k = find(strcmp(imgBases{i}, maskBases), 1);
        if ~isempty(k)
            imgList{end+1,1} = imgListAll{i}; %#ok<AGROW>
            maskList{end+1,1} = maskListAll{k}; %#ok<AGROW>
        end
    end
end

function out = listFilesWithExt(folder, exts)
    out = {};
    for e = exts
        files = dir(fullfile(folder, ['*' e{1}]));
        for k=1:numel(files)
            if ~files(k).isdir
                out{end+1,1} = fullfile(folder, files(k).name); %#ok<AGROW>
            end
        end
    end
end

function s = stripExtension(p)
    [~,n,~] = fileparts(p);
    s = n;
end
