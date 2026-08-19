%% train_acc.m
% The per-epoch history records validation metrics only. This measures the
% selected network's accuracy on a fixed random subset of its own TRAINING
% split, so the article can report train-vs-validation accuracy honestly
% (i.e. quantify the generalisation gap) rather than inferring it.

root   = 'F:\Current_Work\SemanticSegmentationUsingFCN-AlexNet';
out    = 'C:\Users\USER\AppData\Local\Temp\claude\F--Current-Work-SemanticSegmentationUsingFCN-AlexNet\809c6f5e-39a4-4d1b-bc50-83811f54d99e\scratchpad';
folder = fullfile(root,'Improved_Segmentation_Results_transfer');
imgDir = 'F:\Current_Work\Semantic Segmentation Using FCN-AlexNet1\Dataset\Images';
lblDir = 'F:\Current_Work\Semantic Segmentation Using FCN-AlexNet1\Dataset\Lables';

S = load(fullfile(folder,'netSeg_improved.mat'),'netSeg','cfg','trIdx','vaIdx');
netSeg = S.netSeg; cfg = S.cfg;
[imgList, maskList] = pair_images_and_masks(imgDir, lblDir);

CAP = 800;
rng(11);
tr = S.trIdx(randperm(numel(S.trIdx), min(CAP,numel(S.trIdx))));
va = S.vaIdx(randperm(numel(S.vaIdx), min(CAP,numel(S.vaIdx))));

fid = fopen(fullfile(out,'train_acc.txt'),'w');
for which = 1:2
    if which == 1, idx = tr; tag = 'train'; else, idx = va; tag = 'val'; end
    cm = zeros(2,2);
    t0 = tic;
    for i = 1:numel(idx)
        M = imread(maskList{idx(i)});
        if size(M,3) > 1, M = rgb2gray(M); end
        g = M > 127;
        I = im2uint8(imresize(imread(imgList{idx(i)}), cfg.inputSize(1:2)));
        pr = scores2label(minibatchpredict(netSeg, I), cfg.classNames);
        p  = imresize(pr == cfg.classNames(2), size(g), 'nearest');
        cm(1,1) = cm(1,1) + sum(~g(:) & ~p(:));
        cm(1,2) = cm(1,2) + sum(~g(:) &  p(:));
        cm(2,1) = cm(2,1) + sum( g(:) & ~p(:));
        cm(2,2) = cm(2,2) + sum( g(:) &  p(:));
        if mod(i,100)==0, fprintf('%s %d/%d (%.1f min)\n', tag, i, numel(idx), toc(t0)/60); end
    end
    TP = diag(cm); FP = sum(cm,1)' - TP; FN = sum(cm,2) - TP;
    pr_ = TP./(TP+FP+eps); rc_ = TP./(TP+FN+eps);
    f1_ = 2*(pr_.*rc_)./(pr_+rc_+eps);
    iou = TP./(sum(cm,2)+sum(cm,1)'-TP+eps);
    acc = sum(diag(cm))/sum(cm(:));
    fprintf(fid, '%s_n_images = %d\n', tag, numel(idx));
    fprintf(fid, '%s_globalAcc = %.8f\n', tag, acc);
    fprintf(fid, '%s_forged_P = %.8f\n', tag, pr_(2));
    fprintf(fid, '%s_forged_R = %.8f\n', tag, rc_(2));
    fprintf(fid, '%s_forged_F1 = %.8f\n', tag, f1_(2));
    fprintf(fid, '%s_forged_IoU = %.8f\n', tag, iou(2));
    fprintf(fid, '%s_cm = %s\n', tag, mat2str(cm));
    fprintf('%s: acc %.6f  Forged F1 %.6f  IoU %.6f\n', tag, acc, f1_(2), iou(2));
end
fclose(fid);
disp('train/val accuracy measurement complete.');

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
            if ~files(k).isdir, out{end+1,1} = fullfile(folder, files(k).name); end %#ok<AGROW>
        end
    end
end

function s = stripExtension(p)
    [~, n, ~] = fileparts(p); s = n;
end
