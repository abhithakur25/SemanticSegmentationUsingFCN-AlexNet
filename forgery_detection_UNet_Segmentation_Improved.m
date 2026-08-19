%% forgery_detection_UNet_Segmentation_Improved.m
% Improved U-Net segmentation for forged-region localisation.
%
% Motivation (vs forgery_detection_UNet_Segmentation_Final_Grayscale_Tuned.m):
%   The tuned run reached Forged P 0.7329 / R 0.7284 / IoU 0.5756 with train
%   accuracy 99.05% against validation 94.75%. Precision ~= recall means the
%   errors are symmetric, so the bottleneck is not a class-imbalance bias --
%   it is boundary/localisation accuracy plus overfitting. This script targets
%   those two things specifically.
%
% Changes:
%   1. Ported to the R2026a API. unetLayers/deeplabv3plusLayers were REMOVED
%      in R2026a, so the original script no longer runs; this uses unet() +
%      trainnet() + minibatchpredict()/scores2label().
%   2. Geometric augmentation (reflection / rotation / translation / scale)
%      applied identically to image and mask -- addresses the overfit gap.
%   3. Soft Dice loss option, which optimises overlap directly rather than
%      per-pixel likelihood, and weights both classes equally by area.
%   4. Learning-rate decay + early stopping + OutputNetwork='best-validation-loss'.
%   5. Proper 70/15/15 train/val/test split. The original reported its headline
%      metrics on the same split it used for validation monitoring; here the
%      test split is untouched until final evaluation.
%   6. Consistent mask binarisation (>127) for both training labels and
%      evaluation ground truth. The original used Otsu for training labels but
%      >127 for evaluation, which disagreed on 27% of masks (boundary pixels
%      only, but no reason to keep the inconsistency).
%   7. Morphological post-processing, with its parameter chosen on validation
%      and then applied unchanged to test.
%   8. Confusion matrix accumulated incrementally instead of concatenating
%      every pixel into a growing array (the original built a ~137M-element
%      vector pair).
%
% Usage:
%   forgery_detection_UNet_Segmentation_Improved
%   VARIANT='baseline'; forgery_detection_UNet_Segmentation_Improved
%
% VARIANT selects the configuration preset:
%   'improved' (default) - all changes above enabled, from-scratch U-Net
%   'baseline'           - matches the original recipe (CE loss, no augmentation,
%                          12 epochs, constant LR), ported to the new API, for a
%                          controlled A/B on identical splits.
%   'deeplab'            - the 'improved' recipe with a DeepLabV3+ / ResNet-18
%                          backbone instead of the from-scratch U-Net, at a
%                          reduced 1e-4 learning rate suited to fine-tuning
%                          pretrained weights. Requires the ResNet-18 support
%                          package.
%
% All three share cfg.seed, so all three see identical train/val/test splits.

clearvars -except VARIANT CFG_OVERRIDE; close all; clc;

if ~exist('VARIANT','var') || isempty(VARIANT)
    VARIANT = 'improved';
end
fprintf('=== VARIANT: %s ===\n', VARIANT);

%% ---------------- USER CONFIG ----------------
projectRoot = fileparts(mfilename('fullpath'));

% Defaults point at the in-repo SampleData so the script runs out of the box.
% For the real experiment, repoint these at the full Dataset4.
cfg.dataFolder  = fullfile(projectRoot,'SampleData','Images');
cfg.labelFolder = fullfile(projectRoot,'SampleData','Labels');

cfg.inputSize   = [352 480 3];   % EncoderDepth 4 => H,W multiples of 16
cfg.encoderDepth = 4;
cfg.classNames  = ["Background","Forged"];
cfg.miniBatch   = 8;
cfg.seed        = 42;
cfg.splits      = [0.70 0.15 0.15];   % train / val / test

% Caps that only bite on very large datasets. Inf = use the whole split.
cfg.valMonitorCap = Inf;   % val images used for in-training validation
cfg.valTuneCap    = Inf;   % val images used to pick the post-processing minArea
cfg.checkpoint    = false; % save the network every epoch
cfg.verboseEvery  = [];    % [] => once per epoch
cfg.maxImages     = Inf;   % cap dataset size (smoke testing)
cfg.perEpochResults = false; % save a model + metrics after every epoch
cfg.lrDropPeriod  = 10;    % epochs between LR drops (per-epoch loop)
cfg.lrDropFactor  = 0.3;
cfg.patience      = 5;     % early stop after N epochs without Forged-IoU gain

switch lower(VARIANT)
    case 'improved'
        cfg.arch          = 'unet';
        cfg.loss          = 'soft-dice';
        cfg.augment       = true;
        cfg.maxEpochs     = 30;
        cfg.lrSchedule    = 'piecewise';
        cfg.initialLR     = 5e-4;
        cfg.postProcess   = true;
    case 'baseline'
        cfg.arch          = 'unet';
        cfg.loss          = 'crossentropy';
        cfg.augment       = false;
        cfg.maxEpochs     = 12;
        cfg.lrSchedule    = 'none';
        cfg.initialLR     = 5e-4;
        cfg.postProcess   = false;
    case 'deeplab'
        cfg.arch          = 'deeplabv3plus-resnet18';
        cfg.loss          = 'soft-dice';
        cfg.augment       = true;
        cfg.maxEpochs     = 30;
        cfg.lrSchedule    = 'piecewise';
        % 1e-4 rather than the 5e-4 used by 'improved'. 5e-4 is aggressive for
        % an ImageNet-pretrained backbone and can wash out the initialisation
        % in the first few epochs, which would defeat the point of using it.
        % NOTE: this means 'improved' vs 'deeplab' now differs in TWO ways
        % (architecture and learning rate), so that delta no longer isolates
        % the backbone cleanly. Rerun deeplab at 5e-4 if a controlled
        % architecture-only comparison is needed.
        cfg.initialLR     = 1e-4;
        cfg.postProcess   = true;
    case 'transfer'
        % Transfer learning onto the full 47,824-pair Dataset, starting from
        % the ImageNet-pretrained ResNet-18 backbone.
        cfg.dataFolder    = 'G:\Current_Work\Semantic Segmentation Using FCN-AlexNet1\Dataset\Images';
        cfg.labelFolder   = 'G:\Current_Work\Semantic Segmentation Using FCN-AlexNet1\Dataset\Lables';  % sic
        cfg.arch          = 'deeplabv3plus-resnet18';
        cfg.loss          = 'dice-ce';
        cfg.augment       = true;
        cfg.maxEpochs     = 10;
        cfg.lrSchedule    = 'piecewise';
        cfg.initialLR     = 1e-4;   % fine-tuning rate for pretrained weights
        cfg.postProcess   = true;
        cfg.perEpochResults = true; % model + metrics saved after every epoch
        cfg.lrDropPeriod  = 5;      % one LR drop at epoch 6 across a 10-epoch run
        cfg.lrDropFactor  = 0.3;
        cfg.patience      = 4;
        % 10 epochs. ~33.5k training images => ~4185 iterations/epoch. Validating on all
        % ~7.2k val images every epoch would add hours over the run, so
        % in-training monitoring and minArea tuning use a fixed 1000-image
        % subset. The FINAL test metrics still use the entire test split.
        cfg.valMonitorCap = 1000;
        cfg.valTuneCap    = 1000;
        cfg.checkpoint    = false;  % redundant: perEpochResults saves each epoch's net
        cfg.verboseEvery  = 200;

    otherwise
        error('Unknown VARIANT "%s". Use ''baseline'', ''improved'', ''deeplab'' or ''transfer''.', VARIANT);
end

cfg.resultsFolder = fullfile(projectRoot, sprintf('Improved_Segmentation_Results_%s', VARIANT));

% Optional caller overrides, e.g. for smoke-testing:
%   CFG_OVERRIDE.maxImages = 200; CFG_OVERRIDE.maxEpochs = 1;
if exist('CFG_OVERRIDE','var') && isstruct(CFG_OVERRIDE)
    ofn = fieldnames(CFG_OVERRIDE);
    for oi = 1:numel(ofn)
        cfg.(ofn{oi}) = CFG_OVERRIDE.(ofn{oi});
        fprintf('override: cfg.%s\n', ofn{oi});
    end
end

binaryOutDir  = fullfile(cfg.resultsFolder,'BinarySamples');
overlayOutDir = fullfile(cfg.resultsFolder,'OverlaySamples');
figDir        = fullfile(cfg.resultsFolder,'Figures');
for d = {cfg.resultsFolder, binaryOutDir, overlayOutDir, figDir}
    if ~exist(d{1},'dir'), mkdir(d{1}); end
end

rng(cfg.seed);

%% ---------------- PAIR IMAGE & MASK FILES ----------------
[imgList, maskList] = pair_images_and_masks(cfg.dataFolder, cfg.labelFolder);
if isempty(imgList)
    error('No matched image-mask pairs found in:\n  %s\n  %s', cfg.dataFolder, cfg.labelFolder);
end
fprintf('Found %d paired images.\n', numel(imgList));

%% ---------------- SPLIT TRAIN / VAL / TEST ----------------
% Fixed seed => 'baseline' and 'improved' see identical splits.
N = numel(imgList);
perm = randperm(N);

% Optional cap, used for smoke-testing the pipeline on a large dataset.
if isfinite(cfg.maxImages) && N > cfg.maxImages
    perm = perm(1:cfg.maxImages);
    N = cfg.maxImages;
    fprintf('Capped to %d images (cfg.maxImages).\n', N);
end

nTr = round(cfg.splits(1)*N);
nVa = round(cfg.splits(2)*N);
if nTr < 1 || nVa < 1 || (N-nTr-nVa) < 1
    error('Dataset too small (%d) for a %g/%g/%g split.', N, cfg.splits);
end
trIdx = perm(1:nTr);
vaIdx = perm(nTr+1:nTr+nVa);
teIdx = perm(nTr+nVa+1:end);
fprintf('Train %d | Val %d | Test %d\n', numel(trIdx), numel(vaIdx), numel(teIdx));

% Subsets of the validation split used during training / for tuning. The
% final test metrics always use the whole test split.
vaMonIdx  = vaIdx(1:min(numel(vaIdx), cfg.valMonitorCap));
vaTuneIdx = vaIdx(1:min(numel(vaIdx), cfg.valTuneCap));
if numel(vaMonIdx) < numel(vaIdx)
    fprintf('In-training validation uses %d of %d val images.\n', numel(vaMonIdx), numel(vaIdx));
end

%% ---------------- DATASTORES ----------------
dsTrain = make_seg_datastore(imgList(trIdx), maskList(trIdx), cfg, cfg.augment);
dsVal   = make_seg_datastore(imgList(vaMonIdx), maskList(vaMonIdx), cfg, false);

%% ---------------- NETWORK ----------------
switch cfg.arch
    case 'unet'
        net = unet(cfg.inputSize, numel(cfg.classNames), 'EncoderDepth', cfg.encoderDepth);

    case 'deeplabv3plus-resnet18'
        net = deeplabv3plus(cfg.inputSize, numel(cfg.classNames), 'resnet18');

        % deeplabv3plus builds its image input layer with Mean=0 / Std=1, i.e.
        % a no-op zscore. The ResNet-18 backbone was pretrained on ImageNet-
        % normalised input, so feeding raw 0-255 pixels would hand it values
        % ~100x larger than it expects and squander the pretrained weights.
        % Copy the real ImageNet statistics off the reference network.
        ref    = imagePretrainedNetwork('resnet18');
        refIn  = ref.Layers(1);
        inName = net.Layers(1).Name;
        newIn  = imageInputLayer(cfg.inputSize, 'Name', inName, ...
                    'Normalization','zscore', ...
                    'Mean', refIn.Mean, ...
                    'StandardDeviation', refIn.StandardDeviation);
        net = replaceLayer(net, inName, newIn);
        net = initialize(net);
        fprintf('DeepLabV3+/ResNet-18: restored ImageNet normalisation (mean %s).\n', ...
                mat2str(squeeze(refIn.Mean)',5));

    otherwise
        error('Unknown arch "%s".', cfg.arch);
end

switch cfg.loss
    case 'crossentropy'
        lossFcn = "crossentropy";
    case 'soft-dice'
        lossFcn = @softDiceLoss;
    case 'dice-ce'
        lossFcn = @diceCELoss;
    otherwise
        error('Unknown loss "%s".', cfg.loss);
end

iterPerEpoch = max(1, floor(numel(trIdx)/cfg.miniBatch));
if isempty(cfg.verboseEvery), verbEvery = iterPerEpoch; else, verbEvery = cfg.verboseEvery; end
optArgs = { 'InitialLearnRate', cfg.initialLR, ...
            'MaxEpochs',        cfg.maxEpochs, ...
            'MiniBatchSize',    cfg.miniBatch, ...
            'Shuffle',          'every-epoch', ...
            'ValidationData',   dsVal, ...
            'ValidationFrequency', iterPerEpoch, ...
            'Plots',            'none', ...
            'Verbose',          true, ...
            'VerboseFrequency', verbEvery, ...
            'ExecutionEnvironment','auto' };

if strcmp(cfg.lrSchedule,'piecewise')
    optArgs = [optArgs, { 'LearnRateSchedule','piecewise', ...
                          'LearnRateDropFactor',0.3, ...
                          'LearnRateDropPeriod',10, ...
                          'ValidationPatience',8, ...
                          'OutputNetwork','best-validation-loss' }];
end

if cfg.checkpoint
    ckptDir = fullfile(cfg.resultsFolder,'Checkpoints');
    if ~exist(ckptDir,'dir'), mkdir(ckptDir); end
    optArgs = [optArgs, { 'CheckpointPath', ckptDir, ...
                          'CheckpointFrequency', 1, ...
                          'CheckpointFrequencyUnit', 'epoch' }];
    fprintf('Checkpointing every epoch -> %s\n', ckptDir);
end

fprintf('Training (%s loss, augment=%d, %d epochs)...\n', cfg.loss, cfg.augment, cfg.maxEpochs);

if ~cfg.perEpochResults
    % ---- single trainnet call (original behaviour) ----
    opts = trainingOptions('adam', optArgs{:});
    tTrain = tic;
    [netSeg, info] = trainnet(dsTrain, net, lossFcn, opts);
    trainSecs = toc(tTrain);
    fprintf('Training finished in %.1f min.\n', trainSecs/60);
    save(fullfile(cfg.resultsFolder,'netSeg_improved.mat'), ...
         'netSeg','info','cfg','trIdx','vaIdx','teIdx','-v7.3');
else
    % ---- epoch-by-epoch: a model AND metrics land after every epoch ----
    % trainnet cannot carry Adam optimiser state across calls, so the moment
    % estimates restart each epoch. With ~4185 iterations per epoch they
    % re-warm within roughly the first 1000 iterations, which is a small price
    % for artefacts that survive an interrupted multi-hour run.
    epochDir = fullfile(cfg.resultsFolder,'PerEpoch');
    if ~exist(epochDir,'dir'), mkdir(epochDir); end
    histCsv = fullfile(cfg.resultsFolder,'PerEpoch_Metrics.csv');

    netSeg   = net;
    info     = struct('PerEpoch', []);
    lr       = cfg.initialLR;
    bestVal  = -inf;
    bestEp   = 0;
    stale    = 0;
    rows     = table();
    tTrain   = tic;

    for ep = 1:cfg.maxEpochs
        if ep > 1 && mod(ep-1, cfg.lrDropPeriod) == 0
            lr = lr * cfg.lrDropFactor;
            fprintf('LR dropped to %.3g at epoch %d.\n', lr, ep);
        end

        epOpts = trainingOptions('adam', ...
            'InitialLearnRate', lr, ...
            'MaxEpochs',        1, ...
            'MiniBatchSize',    cfg.miniBatch, ...
            'Shuffle',          'every-epoch', ...
            'Plots',            'none', ...
            'Verbose',          true, ...
            'VerboseFrequency', verbEvery, ...
            'ExecutionEnvironment','auto');

        tEp = tic;
        [netSeg, epInfo] = trainnet(dsTrain, netSeg, lossFcn, epOpts);
        epTrainSecs = toc(tEp);

        % --- save the model for this epoch ---
        epModel = fullfile(epochDir, sprintf('net_epoch_%02d.mat', ep));
        save(epModel, 'netSeg','cfg','ep','lr','-v7.3');

        % --- evaluate this epoch on the validation monitoring subset ---
        tEv = tic;
        cmE = evaluate_set(netSeg, imgList(vaMonIdx), maskList(vaMonIdx), cfg, 0);
        mE  = metrics_from_cm(cmE);
        evSecs = toc(tEv);

        trLoss = NaN;
        try, trLoss = epInfo.TrainingHistory.Loss(end); catch, end

        rows = [rows; table(ep, lr, trLoss, ...
                    mE.precision(2), mE.recall(2), mE.f1(2), mE.IoU(2), ...
                    mE.precision(1), mE.recall(1), mE.f1(1), mE.IoU(1), ...
                    mE.globalAcc, mean(mE.IoU), epTrainSecs/60, evSecs/60, ...
                'VariableNames', {'Epoch','LR','TrainLoss', ...
                    'Forged_P','Forged_R','Forged_F1','Forged_IoU', ...
                    'Bg_P','Bg_R','Bg_F1','Bg_IoU', ...
                    'GlobalAcc','MeanIoU','TrainMin','EvalMin'})]; %#ok<AGROW>
        writetable(rows, histCsv);
        save(fullfile(epochDir, sprintf('metrics_epoch_%02d.mat', ep)), 'cmE','mE','ep','lr');

        fprintf(['[epoch %2d/%2d] loss %.4f | Forged P %.4f R %.4f F1 %.4f IoU %.4f | ' ...
                 'acc %.4f | %.1f min train, %.1f min eval\n'], ...
                 ep, cfg.maxEpochs, trLoss, mE.precision(2), mE.recall(2), ...
                 mE.f1(2), mE.IoU(2), mE.globalAcc, epTrainSecs/60, evSecs/60);

        % --- track the best epoch by Forged IoU (the objective that matters) ---
        if mE.IoU(2) > bestVal
            bestVal = mE.IoU(2); bestEp = ep; stale = 0;
            save(fullfile(cfg.resultsFolder,'netSeg_improved.mat'), ...
                 'netSeg','info','cfg','trIdx','vaIdx','teIdx','ep','-v7.3');
            fprintf('  -> new best (Forged IoU %.4f), saved as netSeg_improved.mat\n', bestVal);
        else
            stale = stale + 1;
            if stale >= cfg.patience
                fprintf('Early stop: no Forged-IoU improvement for %d epochs.\n', stale);
                break;
            end
        end
    end

    trainSecs = toc(tTrain);
    info.PerEpoch = rows;
    fprintf('Training finished in %.1f min. Best epoch %d (val Forged IoU %.4f).\n', ...
            trainSecs/60, bestEp, bestVal);

    % Restore the best epoch's weights for the final test evaluation.
    if bestEp > 0
        S_ = load(fullfile(epochDir, sprintf('net_epoch_%02d.mat', bestEp)), 'netSeg');
        netSeg = S_.netSeg;
        fprintf('Restored epoch %d for final evaluation.\n', bestEp);
    end
end

%% ---------------- CHOOSE POST-PROCESSING ON VALIDATION ----------------
if cfg.postProcess
    candidates = [0 64 256 1024];   % min connected-component area, pixels
    bestIoU = -inf; bestArea = 0;
    for a = candidates
        cmV = evaluate_set(netSeg, imgList(vaTuneIdx), maskList(vaTuneIdx), cfg, a);
        m = metrics_from_cm(cmV);
        fprintf('  [val] minArea=%5d -> Forged IoU %.4f\n', a, m.IoU(2));
        if m.IoU(2) > bestIoU, bestIoU = m.IoU(2); bestArea = a; end
    end
    fprintf('Selected minArea=%d on validation (Forged IoU %.4f).\n', bestArea, bestIoU);
else
    bestArea = 0;
end

%% ---------------- FINAL EVALUATION ON TEST ----------------
fprintf('Evaluating on held-out test set (%d images)...\n', numel(teIdx));
[cm, sampleInfo] = evaluate_set(netSeg, imgList(teIdx), maskList(teIdx), cfg, bestArea, ...
                                binaryOutDir, overlayOutDir, 5);
M = metrics_from_cm(cm);

T = table(cfg.classNames(:), M.precision, M.recall, M.f1, M.IoU, ...
    'VariableNames', {'Class','Precision','Recall','F1','IoU'});
disp('Per-class pixel metrics (TEST):'); disp(T);
fprintf('Global pixel accuracy: %.4f\n', M.globalAcc);
fprintf('Mean IoU: %.4f\n', mean(M.IoU));

writetable(T, fullfile(cfg.resultsFolder,'PerClass_PixelMetrics.csv'));
save(fullfile(cfg.resultsFolder,'PixelMetrics_improved.mat'), ...
     'cm','M','T','cfg','bestArea','trainSecs','sampleInfo');

fprintf('\nSaved %d sample Binary & Overlay images.\n', sampleInfo.saved);
fprintf('All outputs -> %s\n', cfg.resultsFolder);

%% ---------------- FIGURES (GRAYSCALE) ----------------
fig1 = figure('Color','w','Visible','off');
imagesc(cm); axis equal tight; colormap(gca,gray); colorbar;
xticks(1:2); yticks(1:2);
xticklabels(cellstr(cfg.classNames)); yticklabels(cellstr(cfg.classNames));
xlabel('Predicted'); ylabel('True');
title(sprintf('Pixel Confusion Matrix (test, %s)', VARIANT));
for r=1:2, for c=1:2
    text(c,r,sprintf('%d',cm(r,c)),'HorizontalAlignment','center','Color','w','FontSize',11);
end, end
saveas(fig1, fullfile(figDir,'ConfusionMatrix_counts.png')); close(fig1);

fig2 = figure('Color','w','Visible','off'); hold on; grid on;
plot(1:2, M.precision, '--o','LineWidth',1.6,'Color',[0 0 0]);
plot(1:2, M.recall,    ':s','LineWidth',1.6,'Color',[0.4 0.4 0.4]);
plot(1:2, M.f1,        '-.^','LineWidth',1.6,'Color',[0.6 0.6 0.6]);
xticks(1:2); xticklabels(cellstr(cfg.classNames)); xlim([0.8 2.2]); ylim([0 1]);
xlabel('Class'); ylabel('Metric'); title('Pixel-level Precision / Recall / F1 (test)');
legend({'Precision','Recall','F1'},'Location','best');
saveas(fig2, fullfile(figDir,'Precision_Recall_F1.png')); close(fig2);

fig3 = figure('Color','w','Visible','off'); hold on; grid on;
lossTr = []; lossVa = [];
try, lossTr = info.TrainingHistory.Loss; catch, end
try, lossVa = info.ValidationHistory.Loss; catch, end
if ~isempty(lossTr), plot(lossTr,'--','LineWidth',1.4,'Color',[0 0 0]); end
if ~isempty(lossVa), plot(lossVa,':','LineWidth',1.4,'Color',[0.4 0.4 0.4]); end
title('Training & Validation Loss'); xlabel('Iteration'); ylabel('Loss');
if ~isempty(lossTr) && ~isempty(lossVa), legend({'Train','Val'}); end
saveas(fig3, fullfile(figDir,'TrainVal_Loss.png')); close(fig3);

%% ================= LOCAL FUNCTIONS =================

function ds = make_seg_datastore(imgs, masks, cfg, doAugment)
    imds = imageDatastore(imgs);
    imds.ReadFcn = @(x) im2uint8(imresize(imread(x), cfg.inputSize(1:2)));
    pxds = pixelLabelDatastore(masks, cfg.classNames, [1 2]);
    pxds.ReadFcn = @(x) readMaskCategorical(x, cfg.inputSize(1:2), cfg.classNames);
    ds = combine(imds, pxds);
    if doAugment
        ds = transform(ds, @(data) augmentPair(data, cfg.classNames));
    end
end

function out = readMaskCategorical(p, tsz, classNames)
    % Fixed >127 threshold, matching the evaluation ground truth exactly.
    M = imread(p);
    if size(M,3) > 1, M = rgb2gray(M); end
    if islogical(M), bw = M; else, bw = M > 127; end
    bw = imresize(bw, tsz, 'nearest');
    out = categorical(bw, [false true], classNames);
end

function data = augmentPair(data, classNames)
    % Geometric augmentation applied identically to image and label.
    I = data{1};
    L = data{2};

    if rand > 0.5, I = fliplr(I); L = fliplr(L); end
    if rand > 0.5, I = flipud(I); L = flipud(L); end

    tform = randomAffine2d('Rotation',[-10 10], ...
                           'XTranslation',[-20 20], ...
                           'YTranslation',[-20 20], ...
                           'Scale',[0.9 1.1]);
    rout = affineOutputView(size(I), tform, 'BoundsStyle','centerOutput');

    I = imwarp(I, tform, 'OutputView', rout, 'FillValues', [0 0 0]);

    % Warp the label through its numeric form, nearest-neighbour, filling new
    % border area with Background (=1) so no spurious Forged pixels appear.
    Ln = uint8(L == classNames(2)) + uint8(1);
    Ln = imwarp(Ln, tform, 'OutputView', rout, 'interp','nearest', 'FillValues', 1);
    L  = categorical(Ln, [1 2], classNames);

    data{1} = I;
    data{2} = L;
end

function loss = softDiceLoss(Y, T)
    % Smoothed per-class soft Dice, averaged over classes and batch.
    % Y: HxWxCxB softmax scores. T: HxWxCxB one-hot targets.
    %
    % Each class contributes equally regardless of pixel area, which lifts the
    % small Forged region without the 1/volume^2 class weighting of Sudre et al.
    % That weighting was tried first and rejected: it explodes to ~1e8 when a
    % mask is empty, and empty Forged targets genuinely occur here (SampleData
    % contains an all-background mask, and augmentation can translate a small
    % forged region off-canvas). The +s smoothing keeps the empty case at
    % loss->0 when the prediction is also empty.
    sdims = [1 2];
    s = 1;
    num = 2*sum(Y .* T, sdims) + s;
    den = sum(Y, sdims) + sum(T, sdims) + s;
    loss = mean(1 - num ./ den, 'all');
end

function loss = diceCELoss(Y, T)
    % Dice + cross-entropy. Dice drives region overlap (the IoU/F1 objective);
    % cross-entropy keeps per-pixel gradients informative, which stabilises
    % early training when Dice alone is nearly flat. Standard pairing for
    % imbalanced segmentation and generally stronger than either alone.
    ce = crossentropy(Y, T, 'NormalizationFactor', 'all-elements');
    loss = softDiceLoss(Y, T) + ce;
end

function [cm, sampleInfo] = evaluate_set(netSeg, imgs, masks, cfg, minArea, binDir, ovDir, nSave)
    % Accumulates a 2x2 pixel confusion matrix at FULL ground-truth resolution.
    if nargin < 6, binDir = ''; ovDir = ''; nSave = 0; end
    cm = zeros(2,2);
    saved = 0;
    names = {};

    for i = 1:numel(imgs)
        I_full = imread(imgs{i});
        M_full = imread(masks{i});
        if size(M_full,3) > 1, M_full = rgb2gray(M_full); end
        gtForged = M_full > 127;

        I_res  = im2uint8(imresize(I_full, cfg.inputSize(1:2)));
        scores = minibatchpredict(netSeg, I_res);
        pred   = scores2label(scores, cfg.classNames);
        predForged_res = (pred == cfg.classNames(2));

        predForged = imresize(predForged_res, [size(gtForged,1) size(gtForged,2)], 'nearest');

        if minArea > 0
            predForged = bwareaopen(predForged, minArea);
            predForged = imfill(predForged, 'holes');
        end

        % 2x2 confusion, rows = true (Bg,Fg), cols = predicted (Bg,Fg)
        g = gtForged(:); p = predForged(:);
        cm(1,1) = cm(1,1) + sum(~g & ~p);
        cm(1,2) = cm(1,2) + sum(~g &  p);
        cm(2,1) = cm(2,1) + sum( g & ~p);
        cm(2,2) = cm(2,2) + sum( g &  p);

        if saved < nSave && ~isempty(binDir)
            saved = saved + 1;
            [~, base, ~] = fileparts(imgs{i});
            names{end+1} = base; %#ok<AGROW>

            binImg = uint8(255*ones(size(predForged),'uint8'));
            binImg(predForged) = 0;
            imwrite(binImg, fullfile(binDir, sprintf('Binary_%s.png', base)));

            if size(I_full,3) == 1, I_vis = cat(3,I_full,I_full,I_full); else, I_vis = I_full; end
            overlay = I_vis;
            overlay(repmat(predForged,[1 1 3])) = 0;
            imwrite(overlay, fullfile(ovDir, sprintf('Overlay_%s.png', base)));
        end
    end

    sampleInfo = struct('saved', saved, 'names', {names});
end

function M = metrics_from_cm(cm)
    TP = diag(cm);
    FP = sum(cm,1)' - TP;
    FN = sum(cm,2)  - TP;
    M.precision = TP ./ (TP + FP + eps);
    M.recall    = TP ./ (TP + FN + eps);
    M.f1        = 2*(M.precision.*M.recall) ./ (M.precision + M.recall + eps);
    M.IoU       = TP ./ (sum(cm,2) + sum(cm,1)' - TP + eps);
    M.globalAcc = sum(diag(cm)) / sum(cm(:));
end

function [imgList, maskList] = pair_images_and_masks(imgDir, maskDir)
    % Basename pairing via ismember (O(n log n)). The original nested
    % find/strcmp loop was O(n^2), which is tolerable for a few thousand files
    % but not for the ~48k-pair Dataset, where it is ~2.3e9 string compares.
    imgFiles  = listFilesWithExt(imgDir,  {'.jpg','.jpeg','.png','.tif','.tiff'});
    maskFiles = listFilesWithExt(maskDir, {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'});
    imgBases  = cellfun(@(p)lower(stripExtension(p)), imgFiles,  'UniformOutput', false);
    maskBases = cellfun(@(p)lower(stripExtension(p)), maskFiles, 'UniformOutput', false);

    [tf, loc] = ismember(imgBases, maskBases);
    imgList  = imgFiles(tf);
    maskList = maskFiles(loc(tf));
    imgList  = imgList(:);
    maskList = maskList(:);
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
