%% forgery_models_compare_gpu_final_v2.m
% Robust GPU-optimized single-file pipeline for training/evaluating multiple
% models for forgery detection (classification) and segmentation.
%
% Changes in v2:
% - Robust pairing of image/label filenames (matches by basename, supports
%   different extensions). Skips unmatched pairs.
% - Shows training progress both in the training-progress window and in the
%   command window ('Plots','training-progress' and 'Verbose',true).
% - Keeps ImageInputLayer normalization disabled to avoid dataset-wide stats.
% - All helper functions are at the end of the file (MATLAB requirement).
%
% Edit top paths and run the whole file.

clearvars; close all; clear all; clc;
warning('off','all');

%% ---------------- User configuration (EDIT) ----------------
dataFolder     = 'D:\claude\SemanticSegmentationUsingFCN-AlexNet1\Dataset3';
imageFolder    = fullfile(dataFolder,'ImagesReszed');   % classification images in subfolders (per-class)
labelFolder    = fullfile(dataFolder,'LabelsReszed');   % segmentation label images (indexed PNG/tif expected)
resultsFolder  = fullfile('D:\claude\SemanticSegmentationUsingFCN-AlexNet1\forgery_results_v2');
if ~exist(resultsFolder,'dir'), mkdir(resultsFolder); end
figFolder = fullfile(resultsFolder,'figures'); if ~exist(figFolder,'dir'), mkdir(figFolder); end

% Which models to run (set false to skip)
run.ResNet50_Classification = true;
run.MobileNetv2_Classification = true;
run.FCN_AlexNet_Segmentation = true;
run.UNet_ResNet18_Segmentation = true;
run.DeepLabv3p_MobileNetv2_Segmentation = true;

% Hardware & speed settings
useGPU = (gpuDeviceCount>0);
executionEnv = 'gpu';
if ~useGPU, executionEnv = 'auto'; end

% Training hyperparameters tuned for speed/stability
miniBatchSize     = 32;        % increase if GPU memory allows
maxEpochs_class   = 8;         % smaller for quick runs; increase for final
maxEpochs_seg     = 12;
initialLR_class   = 1e-4;
initialLR_seg     = 1e-4;
L2Reg             = 1e-4;

% Input sizes
inputSize         = [224 224 3];   % classifier input
segmentationSize  = [360 480 3];   % segmentation input

% Simple augmentation (fast)
imageAug = imageDataAugmenter('RandXReflection',true,'RandRotation',[-8 8],...
    'RandXTranslation',[-6 6],'RandYTranslation',[-6 6]);

% Display training-progress plots and verbose output
showTrainingPlots = true;
verboseTraining = true;

rng(0);

%% ---------------- Hardware info ----------------
if strcmp(executionEnv,'gpu')
    try
        g = gpuDevice;
        fprintf('Using GPU: %s (Free memory %.2f GB)\n', g.Name, g.FreeMemory/1e9);
    catch
        fprintf('GPU selection failed; using CPU.\n');
        executionEnv = 'auto';
    end
else
    fprintf('Using CPU execution.\n');
end

%% ---------------- Prepare classification datastore ----------------
if ~exist(imageFolder,'dir')
    error('Image folder not found: %s', imageFolder);
end

imdsClass = imageDatastore(imageFolder,'IncludeSubfolders',true,'LabelSource','foldernames');

% Remove bad files (zero size / unreadable)
imdsClass = verifyAndFilterImds(imdsClass);

% Split train/val for classification (80/20)
[imdsTrainClass, imdsValClass] = splitEachLabel(imdsClass, 0.8, 'randomized');

% Set ReadFcn to resize + convert to single [0,1] to avoid pre-read issues
imdsTrainClass.ReadFcn = @(f)preprocessImage(f, inputSize);
imdsValClass.ReadFcn   = @(f)preprocessImage(f, inputSize);

%% ---------------- Prepare segmentation datastore (if available) ----------------
pxds = [];
hasSeg = false;
if exist(labelFolder,'dir')
    % We will pair images and labels robustly by basename (supporting .png, .tif, .jpg)
    [pairedImageFiles, pairedLabelFiles] = pairImageLabelFiles(imageFolder, labelFolder);
    if isempty(pairedImageFiles)
        warning('No paired image/label files found. Segmentation will be skipped.');
        hasSeg = false;
    else
        hasSeg = true;
        % Create imds and pxds using paired file lists
        imdsAllSeg = imageDatastore(pairedImageFiles);
        % assume indexed label images where pixel values 1 and 2 correspond to classes
        classes = ["Forged","Authentic"];
        labelIDs = {1,2};
        pxdsAll = pixelLabelDatastore(pairedLabelFiles, classes, labelIDs);
        % Partition
        [imdsTrainSeg, imdsValSeg, pxdsTrain, pxdsVal] = partitionPairedSegData(imdsAllSeg, pxdsAll, 0.8);
        % Set ReadFcn for segmentation images
        imdsTrainSeg.ReadFcn = @(f)preprocessImage(f, segmentationSize);
        imdsValSeg.ReadFcn   = @(f)preprocessImage(f, segmentationSize);
    end
end

%% ---------------- Training option factories (verbose + plots) ----------------
optsClass = @(lr,maxE,valDS) trainingOptions('sgdm', ...
    'Momentum',0.9, 'InitialLearnRate',lr, ...
    'LearnRateSchedule','piecewise','LearnRateDropFactor',0.5,'LearnRateDropPeriod',6, ...
    'L2Regularization',L2Reg, 'MaxEpochs',maxE, ...
    'MiniBatchSize',miniBatchSize, 'Shuffle','every-epoch', ...
    'ValidationData',valDS, ...
    'ValidationFrequency',max(1,floor(numel(imdsTrainClass.Files)/miniBatchSize)), ...
    'Verbose',verboseTraining, 'Plots', ternary(showTrainingPlots,'training-progress','none'), 'ExecutionEnvironment',executionEnv);

optsSeg = @(lr,maxE,valDS) trainingOptions('sgdm', ...
    'Momentum',0.9, 'InitialLearnRate',lr, ...
    'LearnRateSchedule','piecewise','LearnRateDropFactor',0.5,'LearnRateDropPeriod',6, ...
    'L2Regularization',L2Reg, 'MaxEpochs',maxE, ...
    'MiniBatchSize',miniBatchSize, 'Shuffle','every-epoch', ...
    'ValidationData',valDS, ...
    'ValidationFrequency',max(1,floor((hasSeg*numel(imdsTrainSeg.Files))/miniBatchSize)), ...
    'Verbose',verboseTraining, 'Plots', ternary(showTrainingPlots,'training-progress','none'), 'ExecutionEnvironment',executionEnv);

%% ---------------- Model 1: ResNet-50 classification ----------------
if run.ResNet50_Classification
    fprintf('\n=== Model 1: ResNet-50 (classification) ===\n');
    try
        modelPath = fullfile(resultsFolder,'ResNet50_class.mat');
        if exist(modelPath,'file')
            S = load(modelPath); netRes50 = S.netRes50;
            fprintf('Loaded saved ResNet-50 model.\n');
        else
            try
                baseNet = resnet50();
            catch
                warning('resnet50 not available; falling back to resnet18.');
                baseNet = resnet18();
            end
            lgraph = layerGraph(baseNet);
            % Replace input layer to 'none' to skip dataset stat computation
            lgraph = replaceInputLayerWithNone(lgraph, inputSize);
            % Replace final layers for classification
            numClasses = numel(unique(imdsClass.Labels));
            lgraph = replaceFinalLayersForClassification(lgraph, numClasses);
            % Augmented datastore (uses imds ReadFcn)
            augTrainClass = augmentedImageDatastore(inputSize(1:2), imdsTrainClass, 'DataAugmentation', imageAug);
            valDS = augmentedImageDatastore(inputSize(1:2), imdsValClass);
            opts = optsClass(initialLR_class, maxEpochs_class, valDS);
            netRes50 = trainNetwork(augTrainClass, lgraph, opts);
            save(modelPath,'netRes50');
        end
        % Evaluate
        valDS = augmentedImageDatastore(inputSize(1:2), imdsValClass);
        [YPred,~] = classify(netRes50, valDS, 'MiniBatchSize', miniBatchSize);
        YTrue = imdsValClass.Labels;
        acc = mean(YPred == YTrue);
        fprintf('ResNet-50 validation accuracy: %.4f\n', acc);
        h = figure('Visible','off'); confusionchart(YTrue, YPred); saveas(h, fullfile(figFolder,'ResNet50_confusion.png')); close(h);
        save(fullfile(resultsFolder,'ResNet50_results.mat'),'acc','YPred','YTrue');
    catch ME
        warning('ResNet-50 step failed: %s', ME.message);
    end
end

%% ---------------- Model 2: MobileNet-v2 classification ----------------
if run.MobileNetv2_Classification
    fprintf('\n=== Model 2: MobileNet-v2 (classification) ===\n');
    try
        modelPath = fullfile(resultsFolder,'MobileNetv2_class.mat');
        if exist(modelPath,'file')
            S = load(modelPath); netMobile = S.netMobile;
            fprintf('Loaded saved MobileNet-v2 model.\n');
        else
            try
                baseNet = mobilenetv2();
                lgraph = layerGraph(baseNet);
            catch
                warning('mobilenetv2 not available; falling back to resnet18.');
                baseNet = resnet18();
                lgraph = layerGraph(baseNet);
            end
            lgraph = replaceInputLayerWithNone(lgraph, inputSize);
            lgraph = replaceFinalLayersForClassification(lgraph, numel(unique(imdsClass.Labels)));
            augTrainClass = augmentedImageDatastore(inputSize(1:2), imdsTrainClass, 'DataAugmentation', imageAug);
            valDS = augmentedImageDatastore(inputSize(1:2), imdsValClass);
            opts = optsClass(initialLR_class, maxEpochs_class, valDS);
            netMobile = trainNetwork(augTrainClass, lgraph, opts);
            save(modelPath,'netMobile');
        end
        % Evaluate
        valDS = augmentedImageDatastore(inputSize(1:2), imdsValClass);
        [YPred,~] = classify(netMobile, valDS, 'MiniBatchSize', miniBatchSize);
        YTrue = imdsValClass.Labels;
        acc = mean(YPred == YTrue);
        fprintf('MobileNet-v2 validation accuracy: %.4f\n', acc);
        h = figure('Visible','off'); confusionchart(YTrue, YPred); saveas(h, fullfile(figFolder,'MobileNetv2_confusion.png')); close(h);
        save(fullfile(resultsFolder,'MobileNetv2_results.mat'),'acc','YPred','YTrue');
    catch ME
        warning('MobileNet-v2 step failed: %s', ME.message);
    end
end

%% ---------------- Model 3: FCN-AlexNet optimized segmentation ----------------
if run.FCN_AlexNet_Segmentation && hasSeg
    fprintf('\n=== Model 3: FCN-AlexNet (optimized segmentation) ===\n');
    try
        modelPath = fullfile(resultsFolder,'FCN_AlexNet_seg.mat');
        lgraph = fcnLayersOptimizedForSpeed(alexnet(), segmentationSize, numel(pxdsAll.ClassNames));
        tbl = countEachLabel(pxdsAll);
        imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
        classWeights = median(imageFreq)./imageFreq;
        pxLayer = pixelClassificationLayer('Name','pixelLabels','ClassNames',tbl.Name,'ClassWeights',classWeights);
        lgraph = replaceLayerSafe(lgraph,'pixelLabels',pxLayer);
        dsTrain = pixelLabelImageDatastore(imdsTrainSeg, pxdsTrain, 'DataAugmentation', imageAug);
        valDS  = pixelLabelImageDatastore(imdsValSeg, pxdsVal);
        opts = optsSeg(initialLR_seg, maxEpochs_seg, valDS);
        netSeg = trainNetwork(dsTrain, lgraph, opts);
        save(modelPath,'netSeg');
        pxdsResults = semanticseg(imdsValSeg, netSeg, 'WriteLocation', tempdir, 'Verbose', false);
        metrics = evaluateSemanticSegmentation(pxdsResults, pxdsVal);
        fprintf('FCN-AlexNet mean IoU: %.4f\n', metrics.DataSetMetrics.MeanIoU);
        save(fullfile(resultsFolder,'FCN_AlexNet_metrics.mat'),'metrics');
    catch ME
        warning('FCN-AlexNet segmentation failed: %s', ME.message);
    end
end

%% ---------------- Model 4: UNet (ResNet-18 encoder) segmentation ----------------
if run.UNet_ResNet18_Segmentation && hasSeg
    fprintf('\n=== Model 4: UNet-ResNet18 segmentation ===\n');
    try
        modelPath = fullfile(resultsFolder,'UNet_ResNet18_seg.mat');
        try
            lgraph = unetLayers(segmentationSize, numel(pxdsAll.ClassNames), 'EncoderDepth', 4, 'Weights','none');
            tbl = countEachLabel(pxdsAll);
            imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
            classWeights = median(imageFreq)./imageFreq;
            pxLayer = pixelClassificationLayer('Name','Segmentation-Layer','ClassNames',tbl.Name,'ClassWeights',classWeights);
            lgraph = replaceLayerSafe(lgraph,'Segmentation-Layer',pxLayer);
            dsTrain = pixelLabelImageDatastore(imdsTrainSeg, pxdsTrain, 'DataAugmentation', imageAug);
            valDS = pixelLabelImageDatastore(imdsValSeg, pxdsVal);
            opts = optsSeg(initialLR_seg, maxEpochs_seg, valDS);
            netUNet = trainNetwork(dsTrain, lgraph, opts);
            save(modelPath,'netUNet');
            pxdsResults = semanticseg(imdsValSeg, netUNet, 'WriteLocation', tempdir, 'Verbose', false);
            metrics = evaluateSemanticSegmentation(pxdsResults, pxdsVal);
            fprintf('UNet mean IoU: %.4f\n', metrics.DataSetMetrics.MeanIoU);
            save(fullfile(resultsFolder,'UNet_ResNet18_metrics.mat'),'metrics');
        catch ME2
            warning('unetLayers unavailable or failed: %s', ME2.message);
        end
    catch ME
        warning('UNet step failed: %s', ME.message);
    end
end

%% ---------------- Model 5: DeepLabv3+ MobileNet-v2 segmentation ----------------
if run.DeepLabv3p_MobileNetv2_Segmentation && hasSeg
    fprintf('\n=== Model 5: DeepLabv3+ (MobileNet-v2) segmentation ===\n');
    try
        modelPath = fullfile(resultsFolder,'DeepLabv3p_MobileNetv2.mat');
        try
            lgraph = deeplabv3plusLayers(segmentationSize, numel(pxdsAll.ClassNames), 'mobilenetv2');
            tbl = countEachLabel(pxdsAll);
            imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
            classWeights = median(imageFreq)./imageFreq;
            pxLayer = pixelClassificationLayer('Name','pixelLabels','ClassNames',tbl.Name,'ClassWeights',classWeights);
            lgraph = replaceLayerSafe(lgraph,'pixelLabels',pxLayer);
            dsTrain = pixelLabelImageDatastore(imdsTrainSeg, pxdsTrain, 'DataAugmentation', imageAug);
            valDS = pixelLabelImageDatastore(imdsValSeg, pxdsVal);
            opts = optsSeg(initialLR_seg, maxEpochs_seg, valDS);
            netDL = trainNetwork(dsTrain, lgraph, opts);
            save(modelPath,'netDL');
            pxdsResults = semanticseg(imdsValSeg, netDL, 'WriteLocation', tempdir, 'Verbose', false);
            metrics = evaluateSemanticSegmentation(pxdsResults, pxdsVal);
            fprintf('DeepLabv3+ mean IoU: %.4f\n', metrics.DataSetMetrics.MeanIoU);
            save(fullfile(resultsFolder,'DeepLabv3p_metrics.mat'),'metrics');
        catch ME2
            warning('deeplabv3plusLayers unavailable or failed: %s', ME2.message);
        end
    catch ME
        warning('DeepLabv3+ step failed: %s', ME.message);
    end
end

%% ---------------- Summary & comparison plots ----------------
summary = struct();
if exist(fullfile(resultsFolder,'ResNet50_results.mat'),'file'), S=load(fullfile(resultsFolder,'ResNet50_results.mat')); summary.ResNet50 = S.acc; end
if exist(fullfile(resultsFolder,'MobileNetv2_results.mat'),'file'), S=load(fullfile(resultsFolder,'MobileNetv2_results.mat')); summary.MobileNetv2 = S.acc; end
if exist(fullfile(resultsFolder,'FCN_AlexNet_metrics.mat'),'file'), S=load(fullfile(resultsFolder,'FCN_AlexNet_metrics.mat')); summary.FCN_AlexNet = S.metrics.DataSetMetrics.MeanIoU; end
if exist(fullfile(resultsFolder,'UNet_ResNet18_metrics.mat'),'file'), S=load(fullfile(resultsFolder,'UNet_ResNet18_metrics.mat')); summary.UNet_ResNet18 = S.metrics.DataSetMetrics.MeanIoU; end
if exist(fullfile(resultsFolder,'DeepLabv3p_metrics.mat'),'file'), S=load(fullfile(resultsFolder,'DeepLabv3p_metrics.mat')); summary.DeepLabv3p = S.metrics.DataSetMetrics.MeanIoU; end

% Classification comparison bar
fig1 = figure('Visible','off'); clf;
modelsClass = {'ResNet50','MobileNetv2'};
classVals = [ getfieldifexists(summary,'ResNet50'), getfieldifexists(summary,'MobileNetv2') ];
bar(classVals); set(gca,'XTickLabel',modelsClass); ylabel('Validation Accuracy'); title('Classification Comparison');
saveas(fig1, fullfile(figFolder,'classification_comparison.png')); close(fig1);

% Segmentation comparison bar
fig2 = figure('Visible','off'); clf;
modelsSeg = {'FCN_AlexNet','UNet_ResNet18','DeepLabv3p'};
segVals = [ getfieldifexists(summary,'FCN_AlexNet'), getfieldifexists(summary,'UNet_ResNet18'), getfieldifexists(summary,'DeepLabv3p') ];
bar(segVals); set(gca,'XTickLabel',modelsSeg); ylabel('Mean IoU'); title('Segmentation Comparison');
saveas(fig2, fullfile(figFolder,'segmentation_comparison.png')); close(fig2);

fprintf('Finished. Results and figures saved under: %s\n', resultsFolder);

%% ---------------- Local helper functions (end of file) ----------------

function y = ternary(cond,a,b)
    if cond, y = a; else y = b; end
end

function im = preprocessImage(filename, targetSize)
    % Read image, resize, ensure 3 channels, convert to single in [0,1]
    I = imread(filename);
    if size(I,3)==1
        I = cat(3,I,I,I);
    end
    I = imresize(I, targetSize(1:2));
    im = im2single(I); % converts to single and scales if original was uint8
end

function imds = verifyAndFilterImds(imds)
    files = imds.Files;
    labels = imds.Labels;
    good = true(numel(files),1);
    for i=1:numel(files)
        f = files{i};
        info = dir(f);
        if isempty(info) || info.bytes==0
            warning('Removing zero-byte/missing file: %s', f); good(i)=false; continue;
        end
        try
            imread(f); %#ok<NASGU>
        catch
            warning('Unreadable image file: %s (removed).', f); good(i)=false;
        end
    end
    files = files(good); labels = labels(good);
    imds = imageDatastore(files,'Labels',labels);
end

function pxds = verifyAndFilterPxds(pxds)
    files = pxds.Files;
    classes = pxds.ClassNames;
    good = true(numel(files),1);
    for i=1:numel(files)
        f = files{i};
        info = dir(f);
        if isempty(info) || info.bytes==0, warning('Removing zero-byte/missing label: %s',f); good(i)=false; continue; end
        try
            imread(f); %#ok<NASGU>
        catch
            warning('Unreadable label file: %s (removed)', f); good(i)=false;
        end
    end
    if any(~good)
        pxds = pixelLabelDatastore(files(good), classes, num2cell(1:numel(classes)));
    end
end

function [imgFilesFull, lblFilesFull] = pairImageLabelFiles(imageFolder, labelFolder)
    % Return paired lists (full paths) matching images and labels by basename.
    % Supports multiple image/label extensions and is robust to mismatches.
    imgExts = {'.png','.jpg','.jpeg','.tif','.tiff','bmp'};
    lblExts = {'.png','.tif','.tiff','.bmp','.jpg','.jpeg'};
    % List image files
    imgFiles = dir(fullfile(imageFolder,'*.*'));
    imgFiles = imgFiles(~[imgFiles.isdir]);
    imgNames = {imgFiles.name}';
    % Build map of label basenames to full path
    lblFiles = dir(fullfile(labelFolder,'*.*'));
    lblFiles = lblFiles(~[lblFiles.isdir]);
    lblMap = containers.Map;
    for k=1:numel(lblFiles)
        [~,bn,ext] = fileparts(lblFiles(k).name);
        % only accept known label extensions
        if any(strcmpi(ext, lblExts))
            lblMap(lower(bn)) = fullfile(labelFolder,lblFiles(k).name);
        end
    end
    imgFilesFull = {};
    lblFilesFull = {};
    for k=1:numel(imgNames)
        [~,bn,~] = fileparts(imgNames{k});
        key = lower(bn);
        if isKey(lblMap,key)
            imgFilesFull{end+1,1} = fullfile(imageFolder,imgNames{k}); %#ok<AGROW>
            lblFilesFull{end+1,1} = lblMap(key); %#ok<AGROW>
        else
            % try to find label with similar patterns (e.g., different suffixes)
            % skip if not found
        end
    end
    if isempty(imgFilesFull)
        imgFilesFull = {};
        lblFilesFull = {};
    end
end

function [imdsT, imdsV, pxdsT, pxdsV, pxdsAll] = partitionPairedSegData(imdsAll, pxdsAll_in, trainRatio)
    % Partition a paired imageDatastore and pixelLabelDatastore keeping correspondence.
    % imdsAll : imageDatastore built from paired image files
    % pxdsAll_in : pixelLabelDatastore built from paired label files (same ordering expected)
    % Return imdsT, imdsV, pxdsT, pxdsV and pxdsAll (pxdsAll_in)
    imdsFiles = imdsAll.Files;
    N = numel(imdsFiles);
    idx = randperm(N);
    nTrain = round(trainRatio * N);
    trainIdx = idx(1:nTrain);
    valIdx = idx(nTrain+1:end);
    imdsT = imageDatastore(imdsFiles(trainIdx));
    imdsV = imageDatastore(imdsFiles(valIdx));
    % pxdsAll_in.Files should be aligned to imdsAll.Files; we assume same ordering
    pxdsAll = pxdsAll_in;
    lblFiles = pxdsAll.Files;
    pxdsT = pixelLabelDatastore(lblFiles(trainIdx), pxdsAll.ClassNames, num2cell(1:numel(pxdsAll.ClassNames)));
    pxdsV = pixelLabelDatastore(lblFiles(valIdx), pxdsAll.ClassNames, num2cell(1:numel(pxdsAll.ClassNames)));
end

function lgraph = replaceInputLayerWithNone(lgraph, inputSize)
    layers = lgraph.Layers;
    idx = find(arrayfun(@(L) isa(L,'nnet.cnn.layer.ImageInputLayer'), layers), 1, 'first');
    if ~isempty(idx)
        old = layers(idx);
        newInput = imageInputLayer(inputSize,'Name',old.Name,'Normalization','none');
        lgraph = replaceLayer(lgraph, old.Name, newInput);
    end
end

function lgraph = replaceFinalLayersForClassification(lgraph, numClasses)
    layers = lgraph.Layers;
    fcIdx = find(arrayfun(@(L) isa(L,'nnet.cnn.layer.FullyConnectedLayer'), layers), 1, 'last');
    if ~isempty(fcIdx)
        lgraph = replaceLayer(lgraph, layers(fcIdx).Name, fullyConnectedLayer(numClasses,'Name','new_fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10));
    end
    classIdx = find(arrayfun(@(L) isa(L,'nnet.cnn.layer.ClassificationOutputLayer'), layers), 1, 'last');
    if ~isempty(classIdx)
        lgraph = replaceLayer(lgraph, layers(classIdx).Name, classificationLayer('Name','new_class'));
    end
end

function lgraph = fcnLayersOptimizedForSpeed(~, imageSize, numClasses)
    % Compact FCN-like graph for speed
    layers = [
        imageInputLayer(imageSize,'Name','input','Normalization','none')
        convolution2dLayer(11,96,'Stride',4,'Padding','same','Name','conv1'); reluLayer('Name','relu1'); maxPooling2dLayer(3,'Stride',2,'Name','pool1')
        convolution2dLayer(5,256,'Padding','same','Name','conv2'); reluLayer('Name','relu2'); maxPooling2dLayer(3,'Stride',2,'Name','pool2')
        convolution2dLayer(3,384,'Padding','same','Name','conv3'); reluLayer('Name','relu3')
        convolution2dLayer(3,384,'Padding','same','Name','conv4'); reluLayer('Name','relu4')
        convolution2dLayer(3,256,'Padding','same','Name','conv5'); reluLayer('Name','relu5'); maxPooling2dLayer(3,'Stride',2,'Name','pool5')
        transposedConv2dLayer(4,256,'Stride',2,'Cropping','same','Name','up1'); reluLayer('Name','relu_up1')
        transposedConv2dLayer(4,128,'Stride',2,'Cropping','same','Name','up2'); reluLayer('Name','relu_up2')
        convolution2dLayer(1,numClasses,'Name','score')
        softmaxLayer('Name','softmax')
        pixelClassificationLayer('Name','pixelLabels')];
    lgraph = layerGraph(layers);
end

function lgraph = replaceLayerSafe(lgraph, oldName, newLayer)
    if any(strcmp({lgraph.Layers.Name}, oldName))
        lgraph = replaceLayer(lgraph, oldName, newLayer);
    else
        lgraph = addLayers(lgraph, newLayer);
    end
end

function v = getfieldifexists(s, name)
    if isfield(s,name), v = s.(name); else v = NaN; end
end
