%% forgery_models_compare.m
% Single-file pipeline: five model variants for forgery detection/classification/segmentation.
% Save as forgery_models_compare.m and run from top to bottom.
% Adjust dataFolder/resultsFolder to your paths.

clearvars; close all; clc;

%% ---------------- User config ----------------
dataFolder = 'D:\claude\SemanticSegmentationUsingFCN-AlexNet1\Dataset2'; % images and labels underneath
imageFolder = fullfile(dataFolder,'ImagesReszed');      % use resized images (you indicated already resized)
labelFolder = fullfile(dataFolder,'LabelsReszed');      % pixel labels for segmentation (categorical or indexed PNGs)
resultsFolder = 'D:\claude\SemanticSegmentationUsingFCN-AlexNet1\forgery_results';
if ~exist(resultsFolder,'dir'), mkdir(resultsFolder); end
figFolder = fullfile(resultsFolder,'figures'); if ~exist(figFolder,'dir'), mkdir(figFolder); end

% Toggle models to train/evaluate
run.ResNet50_Classification = true;
run.MobileNetv2_Classification = true;
run.FCN_AlexNet_Segmentation = true;
run.UNet_ResNet18_Segmentation = true;
run.DeepLabv3p_MobileNetv2_Segmentation = true;

% Training hyperparameters (sensible defaults tuned for speed & stability)
executionEnv = 'gpu';                 % set 'gpu' if available, else 'auto'
miniBatchSize = 16;                   % adjust to GPU memory: 8/16/32
maxEpochs_class = 12;                 % classification epochs
maxEpochs_seg = 20;                   % segmentation epochs (can increase later)
initialLR_class = 1e-4;               % classification LR for fine-tuning
initialLR_seg = 1e-4;                 % segmentation LR
validationFraction = 0.20;

% Image settings
inputSize = [224 224 3];              % classifier input (ResNet/MobileNet standard)
segmentationSize = [360 480 3];       % segmentation target size (as in your code)

% Fix random seed for reproducibility
rng(0);

%% ---------------- Check GPU ----------------
gpuOK = (gpuDeviceCount>0);
if strcmp(executionEnv,'gpu') && ~gpuOK
    warning('No GPU detected. Switching to CPU execution (very slow).');
    executionEnv = 'auto';
end

%% ---------------- Prepare datastores ----------------
% Classification datastores (we'll use folder structure: dataFolder/Classify/<classfolders>)
classifyFolder = fullfile(dataFolder,'Classify'); % expected structure; if not present, we'll create from Labels maybe
hasClassify = exist(classifyFolder,'dir');

if ~hasClassify
    % If classification images are stored under imageFolder and labels encoded in directory names,
    % we attempt to build a simple imageDatastore using parent-folder labels.
    fprintf('Classification folder not found: %s\nAttempting to use imageFolder and infer labels from subfolders if present.\n', classifyFolder);
    classifyFolder = imageFolder;
    if ~exist(classifyFolder,'dir'), error('No classification images folder found. Put per-class images under %s', classifyFolder); end
end

imdsClass = imageDatastore(classifyFolder, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Split classification dataset
[imdsTrainClass, imdsValClass] = splitEachLabel(imdsClass, 1-validationFraction, 'randomized');

% Segmentation datastores
if exist(labelFolder,'dir') && exist(imageFolder,'dir')
    imdsSeg = imageDatastore(imageFolder);
    % We expect pixel label images stored as indexed PNGs whose pixel values map to classes.
    % Create class names and label IDs based on your dataset: Forgeries vs Authentic
    classes = ["Forged","Authentic"];
    % If the label images are 1/2 indexed PNG where 1=Forged,2=Authentic:
    labelIDs = {1, 2};
    try
        pxds = pixelLabelDatastore(labelFolder, classes, labelIDs);
    catch
        % If label files are RGB masks, try to map colors to IDs (user had camvidPixelLabelIDs earlier)
        % We'll attempt to use a helper color map if available
        fprintf('pixelLabelDatastore failed — expecting indexed label images. Please ensure labels are stored as indexed PNGs or adapt labelIDs.\n');
        pxds = [];
    end
else
    pxds = [];
    fprintf('Segmentation labels or images missing. Segmentation models will be skipped.\n');
end

% Partition segmentation dataset
if ~isempty(pxds)
    [imdsTrainSeg, imdsValSeg, pxdsTrain, pxdsVal] = partitionSegData(imdsSeg, pxds, 0.8);
    fprintf('Segmentation training images: %d, validation images: %d\n', numel(imdsTrainSeg.Files), numel(imdsValSeg.Files));
end

%% ---------------- Helper: common trainingOptions ----------------
optsClass = @(lr, maxE) trainingOptions('sgdm', ...
    'Momentum',0.9, 'InitialLearnRate',lr, ...
    'LearnRateSchedule','piecewise','LearnRateDropFactor',0.5,'LearnRateDropPeriod',6, ...
    'L2Regularization',1e-4, 'MaxEpochs',maxE, ...
    'MiniBatchSize',miniBatchSize, 'Shuffle','every-epoch', ...
    'ValidationData',augmentedImageDatastore(inputSize(1:2), imdsValClass), ...
    'ValidationFrequency',max(1,floor(numel(imdsTrainClass.Files)/miniBatchSize)), ...
    'Verbose',true, 'Plots','training-progress','ExecutionEnvironment',executionEnv);

optsSeg = @(lr, maxE) trainingOptions('sgdm', ...
    'Momentum',0.9, 'InitialLearnRate',lr, ...
    'LearnRateSchedule','piecewise','LearnRateDropFactor',0.5,'LearnRateDropPeriod',6, ...
    'L2Regularization',1e-4, 'MaxEpochs',maxE, ...
    'MiniBatchSize',miniBatchSize, 'Shuffle','every-epoch', ...
    'ValidationData',pixelLabelImageDatastore(imdsValSeg, pxdsVal), ...
    'ValidationFrequency',max(1,floor(numel(imdsTrainSeg.Files)/miniBatchSize)), ...
    'Verbose',true, 'Plots','training-progress','ExecutionEnvironment',executionEnv);

%% ---------------- Model 1: ResNet-50 (Classification) ----------------
if run.ResNet50_Classification
    fprintf('\n=== Model 1: ResNet-50 (classification) ===\n');
    modelPath = fullfile(resultsFolder,'ResNet50_class.mat');
    if exist(modelPath,'file')
        S = load(modelPath); netRes50 = S.netRes50;
        fprintf('Loaded existing ResNet-50 model.\n');
    else
        baseNet = resnet50; % requires Deep Learning Toolbox Model for ResNet-50 Network; if not installed, use resnet18 fallback
        lgraph = layerGraph(baseNet);
        % Replace final fully connected & classification layers
        newFc = fullyConnectedLayer(numel(categories(imdsClass.Labels)), 'Name','new_fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
        newClass = classificationLayer('Name','new_classoutput');
        lgraph = replaceLayer(lgraph,'fc1000',newFc);
        lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',newClass);
        % Create augmented datastore for training with required input size
        augTrainClass = augmentedImageDatastore(inputSize(1:2), imdsTrainClass, 'ColorPreprocessing','rescale-zero-one');
        netRes50 = trainNetwork(augTrainClass, lgraph, optsClass(initialLR_class, maxEpochs_class));
        save(modelPath,'netRes50');
    end
    % Evaluate
    [YPred, scores] = classify(netRes50, augmentedImageDatastore(inputSize(1:2), imdsValClass), 'MiniBatchSize', miniBatchSize);
    YTrue = imdsValClass.Labels;
    acc = mean(YPred == YTrue);
    fprintf('ResNet-50 validation accuracy: %.4f\n', acc);
    conf = confusionmat(YTrue, YPred);
    figure('Name','ResNet50 Confusion'); confusionchart(YTrue,YPred); saveas(gcf, fullfile(figFolder,'ResNet50_confusion.png'));
    % Save metrics
    save(fullfile(resultsFolder,'ResNet50_results.mat'),'acc','YPred','YTrue','conf');
end

%% ---------------- Model 2: MobileNet-v2 (Lightweight classification) ----------------
if run.MobileNetv2_Classification
    fprintf('\n=== Model 2: MobileNet-v2 (classification) ===\n');
    modelPath = fullfile(resultsFolder,'MobileNetv2_class.mat');
    if exist(modelPath,'file')
        S = load(modelPath); netMobile = S.netMobile;
        fprintf('Loaded existing MobileNet-v2 model.\n');
    else
        % If MobileNet-v2 available
        try
            baseNet = mobilenetv2();
            lgraph = layerGraph(baseNet);
            % Replace global pooling/FC/classification
            % Names differ; find last FC layer programmatically:
            lastFc = findLayerByType(lgraph,'nnet.cnn.layer.FullyConnectedLayer');
            % replace with new FC
            newFc = fullyConnectedLayer(numel(categories(imdsClass.Labels)), 'Name','new_fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
            lgraph = replaceLayer(lgraph, lastFc.Name, newFc);
            % replace classification layer name (find by type)
            lastClass = findLayerByType(lgraph,'nnet.cnn.layer.ClassificationOutputLayer');
            lgraph = replaceLayer(lgraph, lastClass.Name, classificationLayer('Name','new_class'));
            augTrainClass = augmentedImageDatastore(inputSize(1:2), imdsTrainClass, 'ColorPreprocessing','rescale-zero-one');
            netMobile = trainNetwork(augTrainClass, lgraph, optsClass(initialLR_class, maxEpochs_class));
        catch
            warning('mobilenetv2 unavailable — falling back to resnet18 small transfer.');
            baseNet = resnet18;
            lgraph = layerGraph(baseNet);
            lgraph = replaceLayer(lgraph,'fc1000',fullyConnectedLayer(numel(categories(imdsClass.Labels)),'Name','new_fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10));
            lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',classificationLayer('Name','new_class'));
            augTrainClass = augmentedImageDatastore(inputSize(1:2), imdsTrainClass);
            netMobile = trainNetwork(augTrainClass, lgraph, optsClass(initialLR_class, maxEpochs_class));
        end
        save(modelPath,'netMobile');
    end
    % Evaluate
    [YPred, scores] = classify(netMobile, augmentedImageDatastore(inputSize(1:2), imdsValClass), 'MiniBatchSize', miniBatchSize);
    YTrue = imdsValClass.Labels;
    acc = mean(YPred == YTrue);
    fprintf('MobileNet-v2 validation accuracy: %.4f\n', acc);
    confusionchart(YTrue,YPred); saveas(gcf, fullfile(figFolder,'MobileNetv2_confusion.png'));
    save(fullfile(resultsFolder,'MobileNetv2_results.mat'),'acc','YPred','YTrue');
    close all;
end

%% ---------------- Model 3: FCN-AlexNet (Optimized segmentation) ----------------
if run.FCN_AlexNet_Segmentation && ~isempty(pxds)
    fprintf('\n=== Model 3: FCN-AlexNet (optimized segmentation) ===\n');
    modelPath = fullfile(resultsFolder,'FCN_AlexNet_seg.mat');
    % Build FCN using alexnet pretrained weights but with reduced padding and faster upsampling
    net = alexnet;
    lgraph = fcnLayersOptimized(net, segmentationSize, numel(pxds.ClassNames));
    % Class weighting (balance classes)
    tbl = countEachLabel(pxds);
    imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
    classWeights = median(imageFreq)./imageFreq;
    pxLayer = pixelClassificationLayer('Name','labels','ClassNames',tbl.Name,'ClassWeights',classWeights);
    lgraph = replaceLayer(lgraph,'pixelLabels', pxLayer);
    % Train
    dsTrain = pixelLabelImageDatastore(imdsTrainSeg, pxdsTrain, 'DataAugmentation', imageDataAugmenter('RandXReflection',true));
    opts = optsSeg(initialLR_seg, maxEpochs_seg);
    netSeg = trainNetwork(dsTrain, lgraph, opts);
    save(modelPath,'netSeg');
    % Evaluate quickly on a few val images to compute IoU
    pxdsResults = semanticseg(imdsValSeg, netSeg, 'WriteLocation', tempdir);
    metrics = evaluateSemanticSegmentation(pxdsResults, pxdsVal);
    fprintf('Mean IoU FCN-AlexNet (train variant): %.4f\n', metrics.DataSetMetrics.MeanIoU);
    save(fullfile(resultsFolder,'FCN_AlexNet_metrics.mat'),'metrics');
end

%% ---------------- Model 4: UNet with ResNet-18 encoder (Segmentation) ----------------
if run.UNet_ResNet18_Segmentation && ~isempty(pxds)
    fprintf('\n=== Model 4: UNet (ResNet-18 encoder) segmentation ===\n');
    modelPath = fullfile(resultsFolder,'UNet_ResNet18_seg.mat');
    if exist(modelPath,'file')
        S = load(modelPath); netUNet = S.netUNet; fprintf('Loaded existing UNet model.\n');
    else
        % Build encoder from resnet18
        baseNet = resnet18;
        % Create encoder features for UNet
        encoderDepth = 4;
        lgraph = unetLayers(segmentationSize, numel(pxds.ClassNames), 'EncoderDepth', encoderDepth, 'Network', baseNet);
        % Class weights
        tbl = countEachLabel(pxds);
        imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
        classWeights = median(imageFreq)./imageFreq;
        pxLayer = pixelClassificationLayer('Name','labels','ClassNames',tbl.Name,'ClassWeights',classWeights);
        lgraph = replaceLayer(lgraph,'Segmentation-Layer', pxLayer); % name may differ; replace robustly below if necessary
        % Train with pixelLabelImageDatastore
        dsTrain = pixelLabelImageDatastore(imdsTrainSeg, pxdsTrain, 'DataAugmentation', imageDataAugmenter('RandXReflection',true));
        opts = optsSeg(initialLR_seg, maxEpochs_seg);
        netUNet = trainNetwork(dsTrain, lgraph, opts);
        save(modelPath,'netUNet');
    end
    % Evaluate
    pxdsResults = semanticseg(imdsValSeg, netUNet, 'WriteLocation', tempdir);
    metrics = evaluateSemanticSegmentation(pxdsResults, pxdsVal);
    fprintf('Mean IoU UNet-ResNet18: %.4f\n', metrics.DataSetMetrics.MeanIoU);
    save(fullfile(resultsFolder,'UNet_ResNet18_metrics.mat'),'metrics');
end

%% ---------------- Model 5: DeepLabv3+ with MobileNet-v2 backbone (Segmentation) ----------------
if run.DeepLabv3p_MobileNetv2_Segmentation && ~isempty(pxds)
    fprintf('\n=== Model 5: DeepLabv3+ (MobileNet-v2 backbone) segmentation ===\n');
    modelPath = fullfile(resultsFolder,'DeepLabv3p_MobileNetv2.mat');
    try
        if exist(modelPath,'file')
            S = load(modelPath); netDL = S.netDL; fprintf('Loaded existing DeepLabv3+ model.\n');
        else
            % Try createDeepLabv3PlusNetwork if available (MATLAB R2020b+)
            dsTrain = pixelLabelImageDatastore(imdsTrainSeg, pxdsTrain, 'DataAugmentation', imageDataAugmenter('RandXReflection',true));
            % Construct DeepLabv3+ with mobilenetv2 backbone if function available
            try
                netDL = deeplabv3plusLayers(segmentationSize, numel(pxds.ClassNames), 'mobilenetv2');
                % assign class weights
                tbl = countEachLabel(pxds); imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount; classWeights = median(imageFreq)./imageFreq;
                pxLayer = pixelClassificationLayer('Name','labels','ClassNames',tbl.Name,'ClassWeights',classWeights);
                netDL = replaceLayer(netDL,'pixelLabels',pxLayer);
                opts = optsSeg(initialLR_seg, maxEpochs_seg);
                netDL = trainNetwork(dsTrain, netDL, opts);
                save(modelPath,'netDL');
            catch
                warning('deeplabv3plusLayers not available; skipping DeepLabv3+ model.');
                netDL = [];
            end
        end
        if ~isempty(netDL)
            pxdsResults = semanticseg(imdsValSeg, netDL, 'WriteLocation', tempdir);
            metrics = evaluateSemanticSegmentation(pxdsResults, pxdsVal);
            fprintf('Mean IoU DeepLabv3+ mobilenetv2: %.4f\n', metrics.DataSetMetrics.MeanIoU);
            save(fullfile(resultsFolder,'DeepLabv3p_metrics.mat'),'metrics');
        end
    catch ME
        warning('DeepLabv3+ training failed: %s', ME.message);
    end
end

%% ---------------- Summarize & Plot comparison metrics ----------------
% Try to load metrics/results for all models and produce comparison plots for classification accuracy and segmentation mIoU
summary = struct();

% Classification results
try
    S = load(fullfile(resultsFolder,'ResNet50_results.mat')); summary.ResNet50_Acc = S.acc; end
try
    S = load(fullfile(resultsFolder,'MobileNetv2_results.mat')); summary.MobileNetv2_Acc = S.acc; end

% Segmentation results (Mean IoU)
try
    S = load(fullfile(resultsFolder,'FCN_AlexNet_metrics.mat')); summary.FCN_AlexNet_IoU = S.metrics.DataSetMetrics.MeanIoU; end
try
    S = load(fullfile(resultsFolder,'UNet_ResNet18_metrics.mat')); summary.UNet_ResNet18_IoU = S.metrics.DataSetMetrics.MeanIoU; end
try
    S = load(fullfile(resultsFolder,'DeepLabv3p_metrics.mat')); summary.DeepLabv3p_IoU = S.metrics.DataSetMetrics.MeanIoU; end
end

% Create simple bar plots
clf;
f1 = figure('Name','Classification Accuracy Comparison','Visible','off');
modelsClass = {'ResNet50','MobileNetv2'};
vals = [getfieldifexists(summary,'ResNet50_Acc'), getfieldifexists(summary,'MobileNetv2_Acc')];
bar(vals);
set(gca,'XTickLabel',modelsClass);
ylabel('Validation Accuracy');
title('Classification comparison');
saveas(f1, fullfile(figFolder,'classification_comparison.png'));

f2 = figure('Name','Segmentation mIoU Comparison','Visible','off');
modelsSeg = {'FCN_AlexNet','UNet_ResNet18','DeepLabv3p_MobileNetv2'};
vals2 = [getfieldifexists(summary,'FCN_AlexNet_IoU'), getfieldifexists(summary,'UNet_ResNet18_IoU'), getfieldifexists(summary,'DeepLabv3p_IoU')];
bar(vals2);
set(gca,'XTickLabel',modelsSeg);
ylabel('MeanIoU');
title('Segmentation comparison');
saveas(f2, fullfile(figFolder,'segmentation_comparison.png'));

%% ---------------- Programmatic Block & Flow diagrams ----------------
hb = figure('Visible','off','Position',[100 100 900 400],'Color','w');
annotation('textbox',[0.02 0.7 0.2 0.2],'String','Data Input','EdgeColor','k','HorizontalAlignment','center');
annotation('textbox',[0.28 0.7 0.2 0.2],'String','Preprocessing','EdgeColor','k','HorizontalAlignment','center');
annotation('textbox',[0.54 0.7 0.2 0.2],'String','Model Training','EdgeColor','k','HorizontalAlignment','center');
annotation('textbox',[0.8 0.7 0.18 0.2],'String','Evaluation & Reporting','EdgeColor','k','HorizontalAlignment','center');
annotation('arrow',[0.2 0.28],[0.78 0.78]); annotation('arrow',[0.5 0.54],[0.78 0.78]); annotation('arrow',[0.76 0.8],[0.78 0.78]);
saveas(hb, fullfile(figFolder,'block_diagram.png')); close(hb);

hf = figure('Visible','off','Position',[100 100 900 600],'Color','w');
annotation('textbox',[0.1 0.9 0.8 0.05],'String','Flow: Load -> Preprocess -> Split -> Train (five models) -> Evaluate -> Save Figures','HorizontalAlignment','center','EdgeColor','none');
ys = 0.75; dy = 0.12;
steps = {'Load images & labels','Preprocess/Resize (done)','Split train/val','Train models (ResNet50/MobileNetv2/FCN/UNet/DeepLabv3+)','Evaluate & Save metrics','Plot comparisons'};
for i=1:numel(steps)
    annotation('textbox',[0.2 ys-(i-1)*dy 0.6 0.08],'String',steps{i},'EdgeColor','k','HorizontalAlignment','center');
    if i < numel(steps)
        annotation('arrow',[0.5 0.5],[ys-(i-1)*dy - 0.02, ys - i*dy + 0.02]);
    end
end
saveas(hf, fullfile(figFolder,'flow_diagram.png')); close(hf);

fprintf('All done. Results and figures saved to %s\n', resultsFolder);

%% ---------------- Local helper functions ----------------

function l = findLayerByType(lgraph, typ)
% find first layer of a given class type
layers = lgraph.Layers;
for i=1:numel(layers)
    if isa(layers(i), typ), l = layers(i); return; end
end
error('Layer of type %s not found', typ);
end

function val = getfieldifexists(s, name)
if isfield(s,name), val = s.(name); else val = NaN; end
end

function [imdsT, imdsV, pxdsT, pxdsV] = partitionSegData(imdsAll, pxdsAll, trainRatio)
% Partition by file indices (keeps correspondence)
num = numel(imdsAll.Files);
idx = randperm(num);
nTrain = round(trainRatio * num);
trainIdx = idx(1:nTrain);
valIdx = idx(nTrain+1:end);
imdsT = imageDatastore(imdsAll.Files(trainIdx));
imdsV = imageDatastore(imdsAll.Files(valIdx));
pxdsT = pixelLabelDatastore(pxdsAll.Files(trainIdx), pxdsAll.ClassNames, 1:numel(pxdsAll.ClassNames));
pxdsV = pixelLabelDatastore(pxdsAll.Files(valIdx), pxdsAll.ClassNames, 1:numel(pxdsAll.ClassNames));
end

function lgraph = fcnLayersOptimized(alexNet, imageSize, numClasses)
% Create an FCN-style layer graph from AlexNet weights, but optimized for speed:
% - reduced large padding
% - smaller transposed conv upsampling (two-stage)
% This is a simplified variant of the approach in your script.

layers = alexNet.Layers;
% Modify conv1 to avoid huge padding (use 'same')
conv1 = convolution2dLayer(layers(2).FilterSize, layers(2).NumFilters, 'Padding','same','Stride',layers(2).Stride,'Name','conv1_new');
conv1.Weights = layers(2).Weights;
conv1.Bias = layers(2).Bias;

% Build a small encoder by keeping first few conv blocks
encoder = [
    imageInputLayer(imageSize,'Normalization','zerocenter','Name','input')
    conv1
    reluLayer('Name','relu1')
    maxPooling2dLayer(2,'Stride',2,'Name','pool1')
    
    convolution2dLayer(3,64,'Padding','same','Name','conv2_1'); batchNormalizationLayer('Name','bn2_1'); reluLayer('Name','relu2_1')
    maxPooling2dLayer(2,'Stride',2,'Name','pool2')
    
    convolution2dLayer(3,128,'Padding','same','Name','conv3_1'); batchNormalizationLayer('Name','bn3_1'); reluLayer('Name','relu3_1')
    ];

% Decoder: upsample by factor 4 using two smaller transposed convs
decoder = [
    transposedConv2dLayer(4, 128, 'Stride',2, 'Cropping','same','Name','up1'); reluLayer('Name','relu_up1')
    transposedConv2dLayer(4, 64, 'Stride',2, 'Cropping','same','Name','up2'); reluLayer('Name','relu_up2')
    convolution2dLayer(1, numClasses, 'Name', 'score')
    softmaxLayer('Name','softmax')
    pixelClassificationLayer('Name','pixelLabels')];

lgraph = layerGraph([encoder; decoder]);
end
