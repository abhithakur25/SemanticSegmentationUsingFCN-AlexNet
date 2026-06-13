%% apply_binary_mask_overlay_batch.m
% Batch overlay: pair RGB images with masks and save results into separate folders.
% -------------------------------------------------------------------------

clearvars; close all; clc;

%% ---------- USER CONFIG ----------
dataRootImgs  = 'D:\claude\SemanticSegmentationUsingFCN-AlexNet1\Dataset4\ImagesReszed';
dataRootMasks = 'D:\claude\SemanticSegmentationUsingFCN-AlexNet1\Dataset4\LabelsReszed';
resultsRoot   = 'D:\claude\SemanticSegmentationUsingFCN-AlexNet1\Dataset4\forgery_results_5models';

outFolder = fullfile(resultsRoot,'overlay_batch_results');
maskFolder    = fullfile(outFolder,'mask_results');
overlayFolder = fullfile(outFolder,'overlay_results');
segOnlyFolder = fullfile(outFolder,'segOnly_results');

if ~exist(outFolder,'dir'), mkdir(outFolder); end
if ~exist(maskFolder,'dir'), mkdir(maskFolder); end
if ~exist(overlayFolder,'dir'), mkdir(overlayFolder); end
if ~exist(segOnlyFolder,'dir'), mkdir(segOnlyFolder); end

% Forged patch overlay settings
forgedColor = [255 0 0]; % red
alpha = 0.5;             % transparency

%% ---------- Gather file lists ----------
imgFiles = dir(fullfile(dataRootImgs,'*.png'));
imgFiles = [imgFiles; dir(fullfile(dataRootImgs,'*.jpg'))];
imgFiles = [imgFiles; dir(fullfile(dataRootImgs,'*.jpeg'))];
imgFiles = [imgFiles; dir(fullfile(dataRootImgs,'*.bmp'))];

maskFiles = dir(fullfile(dataRootMasks,'*.*'));
maskFiles = maskFiles(~[maskFiles.isdir]);

fprintf('Found %d RGB images and %d mask files.\n', numel(imgFiles), numel(maskFiles));

%% ---------- Build a lookup for masks by basename ----------
maskMap = containers.Map();
for k=1:numel(maskFiles)
    [bn,~] = strtok(maskFiles(k).name,'.');
    maskMap(lower(bn)) = fullfile(dataRootMasks, maskFiles(k).name);
end

%% ---------- Process each image ----------
for i=1:numel(imgFiles)
    imgPath = fullfile(dataRootImgs, imgFiles(i).name);
    [bn,~] = strtok(imgFiles(i).name,'.');  % base name without extension
    
    % find corresponding mask
    key = lower(bn);
    if ~isKey(maskMap,key)
        fprintf('No mask found for %s, skipping.\n', imgFiles(i).name);
        continue;
    end
    maskPath = maskMap(key);
    
    % ---- Load images ----
    Iorig = imread(imgPath);
    Iorig = ensure3chan(Iorig);
    [hI,wI,~] = size(Iorig);
    
    maskImg = imread(maskPath);
    maskBW = mask_to_binary(maskImg,hI,wI);
    
    % ---- Overlay ----
    I_overlay = blend_overlay(Iorig, maskBW, forgedColor, alpha);
    segOnly   = color_only_mask(maskBW, forgedColor);
    
    % ---- Save results ----
    imwrite(maskBW, fullfile(maskFolder,    [bn '_mask.png']));
    imwrite(I_overlay, fullfile(overlayFolder, [bn '_overlay.png']));
    imwrite(segOnly, fullfile(segOnlyFolder, [bn '_segOnly.png']));
    
    fprintf('Processed %s -> saved results in subfolders.\n', imgFiles(i).name);
end

fprintf('All overlays saved in: %s\n', outFolder);

%% ---------- Helper functions ----------
function I3 = ensure3chan(I)
    if ndims(I)==2
        I3 = cat(3,I,I,I);
    elseif size(I,3)==1
        I3 = cat(3,I(:,:,1), I(:,:,1), I(:,:,1));
    else
        I3 = I;
    end
end

function maskBW = mask_to_binary(maskImg, targetH, targetW)
    if ndims(maskImg)==3
        gray = rgb2gray(maskImg);
    else
        gray = maskImg;
    end
    g = im2double(gray);
    if max(g(:))>1, g = g./max(g(:)); end
    thresh = graythresh(g);
    maskBW = imbinarize(g,thresh);
    if mean(maskBW(:)) > 0.90
        maskBW = ~maskBW; % invert if almost everything selected
    end
    maskBW = imresize(maskBW,[targetH targetW],'nearest');
    maskBW = logical(maskBW);
end

function I_overlay = blend_overlay(Iorig, maskBW, forgedColor, alpha)
    I_d = im2double(Iorig);
    color_d = double(forgedColor)/255;
    maskL = logical(maskBW);
    out_d = I_d;
    for c=1:3
        ch = out_d(:,:,c);
        ch(maskL) = alpha*color_d(c) + (1-alpha)*ch(maskL);
        out_d(:,:,c) = ch;
    end
    I_overlay = im2uint8(out_d);
end

function segOnly = color_only_mask(maskBW, forgedColor)
    color_d = double(forgedColor)/255;
    segOnly_d = zeros([size(maskBW) 3]);
    for c=1:3
        segOnly_d(:,:,c) = color_d(c) * double(maskBW);
    end
    segOnly = im2uint8(segOnly_d);
end
