%% forgery_models_compare_fixed.m
% Fixed single-file pipeline for training/evaluating five models for
% forgery detection/classification/segmentation.

clearvars; close all; clc;

%% ---------------- User config ----------------
dataFolder = 'D:\claude\SemanticSegmentationUsingFCN-AlexNet1\Dataset2';
imageFolder = fullfile(dataFolder,'ImagesReszed');    % classification images (subfolders per class)
labelFolder = fullfile(dataFolder,'LabelsReszed');    % segmentation label images (indexed PNGs expected)
resultsFolder = fullfile('D:\claude\SemanticSegmentationUsingFCN-AlexNet1\forgery_results');
if ~exist(resultsFolder,'dir'), mkdir(resultsFolder); end
figFolder = fullfile(resultsFolder,'figures'); if ~exist(figFolder,'dir'), mkdir(figFolder); end

% Toggle models to run
run.ResNet50_Classification = true;
run.MobileNetv2_Classification = true;
run.FCN_AlexNet_Segmentation = true;
run.UNet_ResNet18_Segmentation = true;
run.DeepLabv3p_MobileNetv2_Segmentation = true;

% Common training params
executionEnv = 'gpu';    
miniBatchSize = 16;
maxEpochs_class = 12;
maxEpochs_seg = 20;
initialLR_class = 1e-4;
initialLR_seg = 1e-4;
inputSize = [224 224 3];       
segmentationSize = [360 480 3]; 

rng(0);

%% ---------------- Check GPU ----------------
if strcmp(executionEnv,'gpu') && gpuDeviceCount==0
    warning('No GPU detected. Switching to CPU execution.');
    executionEnv = 'auto';
end

%% ---------------- Classification Dataset ----------------
imdsClass = imageDatastore(imageFolder,'IncludeSubfolders',true,'LabelSource','foldernames');
imdsClass = verifyAndFilterImds(imdsClass);

[imdsTrainClass, imdsValClass] = splitEachLabel(imdsClass, 0.8, 'randomized');

%% ---------------- Segmentation Dataset ----------------
pxds = [];
if exist(labelFolder,'dir')
    classes = ["Forged","Authentic"];
    labelIDs = {1,2};  
    try
        pxds = pixelLabelDatastore(labelFolder, classes, labelIDs);
        pxds = verifyAndFilterPxds(pxds);
    catch
        warning('Could not create pixelLabelDatastore. Segmentation will be skipped.');
        pxds = [];
    end
end

if ~isempty(pxds)
    [imdsTrainSeg, imdsValSeg, pxdsTrain, pxdsVal] = partitionSegData(imageFolder,labelFolder,0.8);
end

%% ---------------- Training Options ----------------
optsClass = @(lr,maxE,valDS) trainingOptions('sgdm', ...
    'Momentum',0.9,'InitialLearnRate',lr, ...
    'LearnRateSchedule','piecewise','LearnRateDropFactor',0.5,'LearnRateDropPeriod',6, ...
    'L2Regularization',1e-4,'MaxEpochs',maxE, ...
    'MiniBatchSize',miniBatchSize,'Shuffle','every-epoch', ...
    'ValidationData',valDS, ...
    'ValidationFrequency',max(1,floor(numel(imdsTrainClass.Files)/miniBatchSize)), ...
    'Verbose',true,'Plots','training-progress','ExecutionEnvironment',executionEnv);

optsSeg = @(lr,maxE,valDS) trainingOptions('sgdm', ...
    'Momentum',0.9,'InitialLearnRate',lr, ...
    'LearnRateSchedule','piecewise','LearnRateDropFactor',0.5,'LearnRateDropPeriod',6, ...
    'L2Regularization',1e-4,'MaxEpochs',maxE, ...
    'MiniBatchSize',miniBatchSize,'Shuffle','every-epoch', ...
    'ValidationData',valDS, ...
    'ValidationFrequency',max(1,floor((exist('imdsTrainSeg','var')*numel(imdsTrainSeg.Files))/miniBatchSize)), ...
    'Verbose',true,'Plots','training-progress','ExecutionEnvironment',executionEnv);

%% ---------------- Models Training ----------------
% (Keep your ResNet50, MobileNetv2, FCN-AlexNet, UNet, DeepLabv3+ sections here unchanged)
% I am skipping them here for brevity since only the helper was missing.

%% ---------------- Helper Functions ----------------
function imds = verifyAndFilterImds(imds)
files = imds.Files;
good = true(numel(files),1);
for i=1:numel(files)
    f = files{i};
    info = dir(f);
    if isempty(info) || info.bytes==0
        good(i) = false; continue;
    end
    try, imread(f); catch, good(i) = false; end
end
imds = imageDatastore(files(good),'Labels',imds.Labels(good));
end

function pxds = verifyAndFilterPxds(pxds)
files = pxds.Files;
good = true(numel(files),1);
for i=1:numel(files)
    f = files{i};
    info = dir(f);
    if isempty(info) || info.bytes==0, good(i)=false; continue; end
    try, imread(f); catch, good(i)=false; end
end
if any(~good)
    classes = pxds.ClassNames;
    pxds = pixelLabelDatastore(files(good), classes, num2cell(1:numel(classes)));
end
end

function [imdsTrain, imdsVal, pxdsTrain, pxdsVal] = partitionSegData(imgFolder,labelFolder,trainRatio)
imds = imageDatastore(imgFolder);
classes = ["Forged","Authentic"];
labelIDs = {1,2};
pxds = pixelLabelDatastore(labelFolder,classes,labelIDs);

numFiles = numel(imds.Files);
idx = randperm(numFiles);
N = round(trainRatio*numFiles);
trainIdx = idx(1:N); valIdx = idx(N+1:end);

imdsTrain = imageDatastore(imds.Files(trainIdx));
imdsVal   = imageDatastore(imds.Files(valIdx));
pxdsTrain = pixelLabelDatastore(pxds.Files(trainIdx), classes, labelIDs);
pxdsVal   = pixelLabelDatastore(pxds.Files(valIdx), classes, labelIDs);
end

function lgraph = replaceInputLayerWithNone(lgraph,inputSize)
layers = lgraph.Layers;
idx = find(arrayfun(@(L) isa(L,'nnet.cnn.layer.ImageInputLayer'), layers),1);
if ~isempty(idx)
    old = layers(idx);
    newInput = imageInputLayer(inputSize,'Name',old.Name,'Normalization','none');
    lgraph = replaceLayer(lgraph,old.Name,newInput);
end
end

function lgraph = replaceFinalLayersForClassification(lgraph,numClasses)
layers = lgraph.Layers;
fcIdx = find(arrayfun(@(L) isa(L,'nnet.cnn.layer.FullyConnectedLayer'), layers),1,'last');
if ~isempty(fcIdx)
    newFc = fullyConnectedLayer(numClasses,'Name','new_fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
    lgraph = replaceLayer(lgraph,layers(fcIdx).Name,newFc);
end
classIdx = find(arrayfun(@(L) isa(L,'nnet.cnn.layer.ClassificationOutputLayer'), layers),1,'last');
if ~isempty(classIdx)
    lgraph = replaceLayer(lgraph,layers(classIdx).Name,classificationLayer('Name','new_class'));
end
end
