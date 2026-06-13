% Semantic Segmentation Using FCN-AlexNet

clear all, close all, clc;

%% Setup
alexnet();

imgDir = fullfile('D:\claude\SemanticSegmentationUsingFCN-AlexNet1\Dataset2','Images');
imds = imageDatastore(imgDir);

I = readimage(imds, 101);
I = histeq(I);
figure
imshow(I)

%%
classes = [
    "Forged"
    "Authentic"
    ];


labelIDs = camvidPixelLabelIDs();

labelDir = fullfile('D:\claude\SemanticSegmentationUsingFCN-AlexNet1\Dataset2','Labels');
pxds = pixelLabelDatastore(labelDir,classes,labelIDs);

C = readimage(pxds, 101);

cmap = camvidColorMap;
B = labeloverlay(I,C,'ColorMap',cmap);

figure
imshow(B)
pixelLabelColorbar(cmap,classes);


%%
tbl = countEachLabel(pxds)

frequency = tbl.PixelCount/sum(tbl.PixelCount);

figure
bar(1:numel(classes),frequency)
xticks(1:numel(classes)) 
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency')

%%
imageFolder = fullfile('D:\claude\SemanticSegmentationUsingFCN-AlexNet1\Dataset2','ImagesReszed',filesep);
imds = resizeCamVidImages(imds,imageFolder);

labelFolder = fullfile('D:\claude\SemanticSegmentationUsingFCN-AlexNet1\Dataset2','LabelsReszed',filesep);
pxds = resizeCamVidPixelLabels(pxds,labelFolder);

%% Prepare Training and Test Sets

[imdsTrain, imdsTest, pxdsTrain, pxdsTest] = partitionCamVidData(imds,pxds);
%% 

numTrainingImages = numel(imdsTrain.Files)
numTestingImages = numel(imdsTest.Files)

%% Create the Network

imageSize = [360 480];
numClasses = numel(classes);

net = alexnet();

layers = net.Layers;

layers(1) = imageInputLayer([imageSize 3], 'Name', layers(1).Name,...
    'DataAugmentation', layers(1).DataAugmentation, ...
    'Normalization', layers(1).Normalization);


% fc6 is layers 17
idx = 17;
weights = layers(idx).Weights';
weights = reshape(weights, 6, 6, 256, 4096);
bias = reshape(layers(idx).Bias, 1, 1, []);

layers(idx) = convolution2dLayer(6, 4096, 'NumChannels', 256, 'Name', 'fc6');
layers(idx).Weights = weights;
layers(idx).Bias = bias;

% fc7 is layers 20
idx = 20;
weights = layers(idx).Weights';
weights = reshape(weights, 1, 1, 4096, 4096);
bias = reshape(layers(idx).Bias, 1, 1, []);

layers(idx) = convolution2dLayer(1, 4096, 'NumChannels', 4096, 'Name', 'fc7');
layers(idx).Weights = weights;
layers(idx).Bias = bias;


conv1 = layers(2);
conv1New = convolution2dLayer(conv1.FilterSize, conv1.NumFilters, ...
    'Stride', conv1.Stride, ...
    'Padding', [100 100], ...
    'NumChannels', conv1.NumChannels, ...
    'WeightLearnRateFactor', conv1.WeightLearnRateFactor, ...
    'WeightL2Factor', conv1.WeightL2Factor, ...
    'BiasLearnRateFactor', conv1.BiasLearnRateFactor, ...
    'BiasL2Factor', conv1.BiasL2Factor, ...
    'Name', conv1.Name);
conv1New.Weights = conv1.Weights;
conv1New.Bias = conv1.Bias;

layers(2) = conv1New;

layers(end-2:end) = [];

upscore = transposedConv2dLayer(64, numClasses, ...
    'NumChannels', numClasses, 'Stride', 32, 'Name', 'upscore');

layers = [
    layers
    convolution2dLayer(1, numClasses, 'Name', 'score_fr');
    upscore  
    crop2dLayer('centercrop', 'Name', 'score')
    softmaxLayer('Name', 'softmax')
    pixelClassificationLayer('Name', 'pixelLabels')
    ];

lgraph = layerGraph(layers);

lgraph = connectLayers(lgraph, 'data', 'score/ref');

%% Balance Classes Using Class Weighting
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq

classWeights=[1;1]

pxLayer = pixelClassificationLayer('Name','labels','ClassNames', tbl.Name, 'ClassWeights', classWeights)

lgraph = removeLayers(lgraph, 'pixelLabels');
lgraph = addLayers(lgraph, pxLayer);
lgraph = connectLayers(lgraph, 'softmax' ,'labels');

figure, plot(lgraph)

%% Select Training Options
% Show the live training-progress plot only when a MATLAB desktop/GUI is
% available; fall back to 'none' for headless runs (e.g. matlab -batch).
if usejava('desktop')
    plotMode = 'training-progress';
else
    plotMode = 'none';
end

options = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 1e-3, ...
    'L2Regularization', 0.0005, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 10, ...
    'Shuffle', 'every-epoch', ...
    'ExecutionEnvironment','auto', ...
    'Plots', plotMode, ...
    'Verbose', true, ...
    'VerboseFrequency', 10);

%% Data Augmentation
augmenter = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation', [-10 10], 'RandYTranslation',[-10 10]);

%% Start Training

datasource = pixelLabelImageSource(imdsTrain,pxdsTrain,...
    'DataAugmentation',augmenter);

%%

[net, info] = trainNetwork(datasource,lgraph,options);

%% Test Network on One Image
% Pick up to 11 test images spread across the available test set, instead of
% hardcoded indices 2240:2250 (which error if the test split is smaller).
numTest = numel(imdsTest.Files);
sampleIdx = unique(round(linspace(1, numTest, min(11, numTest))));
for idx = sampleIdx
% idx = 6;
I = readimage(imdsTest,idx);
C = semanticseg(I, net);

% imshow: original vs. predicted-label overlay (montage)
B = labeloverlay(I, C, 'Colormap', cmap, 'Transparency',0.4);
hOverlay = figure; imshowpair(I, B, 'montage')
pixelLabelColorbar(cmap, classes);
saveas(hOverlay,fullfile('D:\claude\SemanticSegmentationUsingFCN-AlexNet1\Results',['overlay_' num2str(idx) '.png']));


% predicted vs. ground-truth label maps
expectedResult = readimage(pxdsTest,idx);
actual = uint8(C);
expected = uint8(expectedResult);
hCompare = figure;
imshowpair(actual, expected)
saveas(hCompare,fullfile('D:\claude\SemanticSegmentationUsingFCN-AlexNet1\Results',['compare_' num2str(idx) '.png']));
end
%% 

iou = jaccard(C, expectedResult);
table(classes,iou)

%%
pxdsResults = semanticseg(imdsTest,net,'WriteLocation',tempdir,'Verbose',false);

metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',false);


metrics.DataSetMetrics

metrics.ClassMetrics


function labelIDs = camvidPixelLabelIDs()
labelIDs = { ...
    
    % "Sky"
    [
    250 250 250; ... % "Authentic"
    ]
    
    % "OtherObjects" 
    [
    000 000 000; ... % "Forged"
%     250 250 250; ... % "Authentic
%     128 000 000; ... % "Building"
%     064 192 000; ... % "Wall"
%     064 000 064; ... % "Tunnel"
%     192 000 128; ... % "Archway"
%     192 192 128; ... % "Column_Pole"
%     000 000 064; ... % "TrafficCone"
%     128 064 128; ... % "Road"
%     128 000 192; ... % "LaneMkgsDriv"
%     192 000 064; ... % "LaneMkgsNonDriv"
%     000 000 192; ... % "Sidewalk" 
%     064 192 128; ... % "ParkingBlock"
%     128 128 192; ... % "RoadShoulder"
%     128 128 000; ... % "Tree"
%     192 192 000; ... % "VegetationMisc"
%     192 128 128; ... % "SignSymbol"
%     128 128 064; ... % "Misc_Text"
%     000 064 064; ... % "TrafficLight"
%     064 064 128; ... % "Fence"
%     064 000 128; ... % "Car"
%     064 128 192; ... % "SUVPickupTruck"
%     192 128 192; ... % "Truck_Bus"
%     192 064 128; ... % "Train"
%     128 064 064; ... % "OtherMoving"
%     064 064 000; ... % "Pedestrian"
%     192 128 064; ... % "Child"
%     064 000 192; ... % "CartLuggagePram"
%     064 128 064; ... % "Animal"
%     000 128 192; ... % "Bicyclist"
%     192 000 192; ... % "MotorcycleScooter"
    ]
    
    };
end
%% 

function pixelLabelColorbar(cmap, classNames)

colormap(gca,cmap)

% Add colorbar to current figure.
c = colorbar('peer', gca);

% Use class names for tick marks.
c.TickLabels = classNames;
numClasses = size(cmap,1);

% Center tick labels.
c.Ticks = 1/(numClasses*2):1/numClasses:1;

% Remove tick mark.
c.TickLength = 0;
end
%% 

function cmap = camvidColorMap()
% CamVid�f?[�^�Z�b�g�̊e�N���X�ɑ΂��ĕR�t������?F(colormap)���`���܂�?B
% Define the colormap used by CamVid dataset.

cmap = [
    60 40 222   % Sky
    128 0 0     % OtherObjects
    ];

% Normalize between [0 1].
cmap = cmap ./ 255;
end
%% 
% 

function imds = resizeCamVidImages(imds, imageFolder)
% Resize images to [360 480].

if ~exist(imageFolder,'dir') 
    mkdir(imageFolder)
else
    imds = imageDatastore(imageFolder);
    return; % Skip if images already resized
end

reset(imds)
while hasdata(imds)
    % Read an image.
    [I,info] = read(imds);     
    
    % Resize image.
    I = imresize(I,[360 480]);    
    
    % Write to disk.
    [~, filename, ext] = fileparts(info.Filename);
    imwrite(I,[imageFolder filename ext])
end

imds = imageDatastore(imageFolder);
end
%% 
% 

function pxds = resizeCamVidPixelLabels(pxds, labelFolder)
% Resize pixel label data to [360 480].

classes = pxds.ClassNames;
labelIDs = 1:numel(classes);
if ~exist(labelFolder,'dir')
    mkdir(labelFolder)
else
    pxds = pixelLabelDatastore(labelFolder,classes,labelIDs);
    return; % Skip if images already resized
end

reset(pxds)
while hasdata(pxds)
    % Read the pixel data.
    [C,info] = read(pxds);
    
    % Convert from categorical to uint8.
    L = uint8(C);
    
    % Resize the data. Use 'nearest' interpolation to
    % preserve label IDs.
    L = imresize(L,[360 480],'nearest');
    
    % Write the data to disk.
    [~, filename, ext] = fileparts(info.Filename);
    imwrite(L,[labelFolder filename ext])
end

labelIDs = 1:numel(classes);
pxds = pixelLabelDatastore(labelFolder,classes,labelIDs);
end
%% 
% 

function [imdsTrain, imdsTest, pxdsTrain, pxdsTest] = partitionCamVidData(imds,pxds)
% Partition CamVid data by randomly selecting 60% of the data for training. The
% rest is used for testing.
    
% Set initial random state for example reproducibility.
rng(0); 
numFiles = numel(imds.Files);
shuffledIndices = randperm(numFiles);

% Use 60% of the images for training.
N = round(0.60 * numFiles);
trainingIdx = shuffledIndices(1:N);

% Use the rest for testing.
testIdx = shuffledIndices(N+1:end);

% Create image datastores for training and test.
trainingImages = imds.Files(trainingIdx);
testImages = imds.Files(testIdx);
imdsTrain = imageDatastore(trainingImages);
imdsTest = imageDatastore(testImages);

% Extract class and label IDs info.
classes = pxds.ClassNames;
labelIDs = 1:numel(pxds.ClassNames);

% Create pixel label datastores for training and test.
trainingLabels = pxds.Files(trainingIdx);
testLabels = pxds.Files(testIdx);
pxdsTrain = pixelLabelDatastore(trainingLabels, classes, labelIDs);
pxdsTest = pixelLabelDatastore(testLabels, classes, labelIDs);
end