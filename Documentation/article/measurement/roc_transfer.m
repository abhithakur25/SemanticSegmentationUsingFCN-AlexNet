%% roc_transfer.m
% Threshold-free evaluation of the saved transfer model (DeepLabV3+/ResNet-18)
% on its own held-out test split. Accumulates a 10,000-bin histogram of the
% Forged-class softmax score, separately for ground-truth Forged and
% ground-truth Background pixels, then derives ROC / PR / AUC exactly.
%
% Scores are up-sampled to full ground-truth resolution so that the curves
% describe the same pixel population as the reported confusion matrix.

root    = 'F:\Current_Work\SemanticSegmentationUsingFCN-AlexNet';
out     = 'C:\Users\USER\AppData\Local\Temp\claude\F--Current-Work-SemanticSegmentationUsingFCN-AlexNet\809c6f5e-39a4-4d1b-bc50-83811f54d99e\scratchpad';
folder  = fullfile(root,'Improved_Segmentation_Results_transfer');

% The saved cfg points at G:\ ; this machine now mounts the same tree on F:\.
imgDir  = 'F:\Current_Work\Semantic Segmentation Using FCN-AlexNet1\Dataset\Images';
lblDir  = 'F:\Current_Work\Semantic Segmentation Using FCN-AlexNet1\Dataset\Lables';

fprintf('Loading network...\n');
S = load(fullfile(folder,'netSeg_improved.mat'),'netSeg','cfg','trIdx','vaIdx','teIdx');
netSeg = S.netSeg; cfg = S.cfg; teIdx = S.teIdx;
classNames = cfg.classNames;
fprintf('Test split: %d images\n', numel(teIdx));

% Re-derive the identical file pairing the training script used.
[imgList, maskList] = pair_images_and_masks(imgDir, lblDir);
fprintf('Paired %d files.\n', numel(imgList));

teImgs  = imgList(teIdx);
teMasks = maskList(teIdx);

try
    fprintf('gpuDeviceCount = %d\n', gpuDeviceCount);
catch
    fprintf('gpuDeviceCount unavailable\n');
end

% Threshold-free curves need per-pixel softmax scores, which is far more
% expensive than the argmax pass used for the reported confusion matrix.
% A fixed random subset of the test split is used; at 172,800 px per image
% even 1,200 images give >2e8 scored pixels, which is ample for a stable ROC.
CAP = 1200;
if numel(teImgs) > CAP
    rng(7);
    sub = randperm(numel(teImgs), CAP);
    teImgs = teImgs(sub); teMasks = teMasks(sub);
    fprintf('ROC computed on a fixed random subset of %d/%d test images.\n', CAP, numel(teIdx));
end

NB   = 10000;                 % score histogram bins
hPos = zeros(NB,1);           % GT Forged
hNeg = zeros(NB,1);           % GT Background
cmChk = zeros(2,2);           % re-derived argmax confusion matrix (sanity check)

t0 = tic;
N  = numel(teImgs);
for i = 1:N
    M_full = imread(teMasks{i});
    if size(M_full,3) > 1, M_full = rgb2gray(M_full); end
    g = M_full > 127;

    I_res  = im2uint8(imresize(imread(teImgs{i}), cfg.inputSize(1:2)));
    scores = minibatchpredict(netSeg, I_res);        % H x W x 2
    sF     = double(scores(:,:,2));                  % Forged channel
    sF     = imresize(sF, size(g), 'bilinear');
    sF     = min(max(sF,0),1);

    p = sF > 0.5;
    cmChk(1,1) = cmChk(1,1) + sum(~g(:) & ~p(:));
    cmChk(1,2) = cmChk(1,2) + sum(~g(:) &  p(:));
    cmChk(2,1) = cmChk(2,1) + sum( g(:) & ~p(:));
    cmChk(2,2) = cmChk(2,2) + sum( g(:) &  p(:));

    b = min(NB, max(1, floor(sF*NB) + 1));
    hPos = hPos + accumarray(b(g),  1, [NB 1]);
    hNeg = hNeg + accumarray(b(~g), 1, [NB 1]);

    if mod(i,25) == 0
        fprintf('  %5d/%d  (%.1f min elapsed)\n', i, N, toc(t0)/60);
    end
end
fprintf('Inference done in %.1f min.\n', toc(t0)/60);

P = sum(hPos); Nn = sum(hNeg);

% Sweep the threshold downwards: bin k and above are predicted Forged.
TP = flipud(cumsum(flipud(hPos)));
FP = flipud(cumsum(flipud(hNeg)));
FN = P  - TP;
TN = Nn - FP;

TPR = TP ./ max(P,1);
FPR = FP ./ max(Nn,1);
PRE = TP ./ max(TP + FP, 1);
REC = TPR;
F1  = 2*TP ./ max(2*TP + FP + FN, 1);
MCC = (TP.*TN - FP.*FN) ./ max(sqrt((TP+FP).*(TP+FN).*(TN+FP).*(TN+FN)), 1);

% Prepend the (0,0) operating point (threshold above every score).
FPRc = [0; FPR]; TPRc = [0; TPR];
AUC  = trapz(FPRc, TPRc);
[~, ord] = sort(REC);
AUC_PR = trapz(REC(ord), PRE(ord));

[bestF1, kF1] = max(F1);
[bestMCC, kM] = max(MCC);
thr = ((0:NB-1)' + 0.5) / NB;

sens = TP(round(0.5*NB)) / max(P,1);
spec = TN(round(0.5*NB)) / max(Nn,1);

fprintf('\n=== Threshold-free metrics (test split, %d images, %.0f pixels) ===\n', N, P+Nn);
fprintf('AUC(ROC)          = %.6f\n', AUC);
fprintf('AUC(PR)           = %.6f\n', AUC_PR);
fprintf('Best F1           = %.6f at threshold %.4f\n', bestF1, thr(kF1));
fprintf('Best MCC          = %.6f at threshold %.4f\n', bestMCC, thr(kM));
fprintf('MCC @0.5          = %.6f\n', MCC(round(0.5*NB)));
fprintf('Sensitivity @0.5  = %.6f\n', sens);
fprintf('Specificity @0.5  = %.6f\n', spec);
fprintf('Forged prevalence = %.6f\n', P/(P+Nn));
fprintf('CM check @0.5 (bilinear-upsampled scores):\n');
fprintf('  %d %d\n  %d %d\n', cmChk(1,1),cmChk(1,2),cmChk(2,1),cmChk(2,2));

% Down-sample the curves to 1000 points for plotting.
sel = unique(round(linspace(1, NB, 1000)));
Tbl = table(thr(sel), FPR(sel), TPR(sel), PRE(sel), REC(sel), F1(sel), MCC(sel), ...
    'VariableNames', {'Threshold','FPR','TPR','Precision','Recall','F1','MCC'});
writetable(Tbl, fullfile(out,'roc_transfer_curve.csv'));

summary = struct('N_images',N,'N_pixels',P+Nn,'AUC',AUC,'AUC_PR',AUC_PR, ...
                 'bestF1',bestF1,'bestF1_thr',thr(kF1),'bestMCC',bestMCC, ...
                 'bestMCC_thr',thr(kM),'MCC_at_half',MCC(round(0.5*NB)), ...
                 'sensitivity',sens,'specificity',spec, ...
                 'prevalence',P/(P+Nn),'cm_check',cmChk);
save(fullfile(out,'roc_transfer.mat'),'hPos','hNeg','AUC','AUC_PR','Tbl','summary','cmChk','-v7.3');

fid = fopen(fullfile(out,'roc_transfer_summary.txt'),'w');
fn = fieldnames(summary);
for k = 1:numel(fn)
    v = summary.(fn{k});
    if isscalar(v), fprintf(fid,'%s = %.10f\n', fn{k}, double(v));
    else, fprintf(fid,'%s = %s\n', fn{k}, mat2str(v)); end
end
fclose(fid);
disp('ROC extraction complete.');

%% ---- pairing helper (identical logic to the training script) ----
function [imgList, maskList] = pair_images_and_masks(imgDir, maskDir)
    imgFiles  = listFilesWithExt(imgDir,  {'.jpg','.jpeg','.png','.tif','.tiff'});
    maskFiles = listFilesWithExt(maskDir, {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'});
    imgBases  = cellfun(@(p)lower(stripExtension(p)), imgFiles,  'UniformOutput', false);
    maskBases = cellfun(@(p)lower(stripExtension(p)), maskFiles, 'UniformOutput', false);
    [tf, loc] = ismember(imgBases, maskBases);
    imgList  = imgFiles(tf);  imgList  = imgList(:);
    maskList = maskFiles(loc(tf)); maskList = maskList(:);
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
