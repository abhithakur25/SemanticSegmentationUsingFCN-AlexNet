%% gen_doc_figures.m  — generate confusion matrix, ROC and PR/AUC figures
% Uses the REAL trained U-Net (Final_Segmentation_Results_Tuned\netSeg_final.mat)
% and the REAL saved confusion matrix; runs inference on a subset of Dataset4
% validation images to obtain genuine per-pixel forged-class scores.
clear; clc; close all;
root   = 'D:\claude\SemanticSegmentationUsingFCN-AlexNet1';
outDir = fullfile(root,'Documentation','figures');
if ~exist(outDir,'dir'), mkdir(outDir); end

%% 1) Confusion matrix from the real saved metrics
M = load(fullfile(root,'Final_Segmentation_Results_Tuned','PixelMetrics_final.mat'));
cm = M.cm;                      % [Background;Forged] x [Background;Forged]
cmn = cm ./ sum(cm,2);          % row-normalised
labels = {'Background','Forged'};
f1 = figure('Color','w','Visible','off','Position',[100 100 460 400]);
imagesc(cmn); colormap(flipud(gray)); caxis([0 1]);
set(gca,'XTick',1:2,'XTickLabel',labels,'YTick',1:2,'YTickLabel',labels,'FontSize',11);
xlabel('Predicted class'); ylabel('True class');
title('Pixel-level Confusion Matrix (row-normalised)');
for r=1:2, for c=1:2
    txt = sprintf('%d\n(%.2f%%)', cm(r,c), 100*cmn(r,c));
    if cmn(r,c) > 0.5, col='w'; else, col='k'; end
    text(c,r,txt,'HorizontalAlignment','center','Color',col,'FontSize',11,'FontWeight','bold');
end, end
colorbar;
saveas(f1, fullfile(outDir,'confusion_matrix.png')); close(f1);
fprintf('Confusion matrix figure saved.\n');

%% 2) Run inference on a subset to get real per-pixel forged scores
N = load(fullfile(root,'Final_Segmentation_Results_Tuned','netSeg_final.mat'));
net = N.netSeg;
inSize = net.Layers(1).InputSize;     % [H W C]
imgDir = fullfile(root,'Dataset4','Images');
mskDir = fullfile(root,'Dataset4','Labels');
imgs = [dir(fullfile(imgDir,'*.png')); dir(fullfile(imgDir,'*.jpg')); dir(fullfile(imgDir,'*.jpeg'))];
rng(42); sel = randperm(numel(imgs), min(40,numel(imgs)));
allScore = []; allLabel = [];
classNames = string(net.Layers(end).Classes);
forgedIdx = find(classNames=="Forged"); if isempty(forgedIdx), forgedIdx = 2; end
cnt = 0;
for k = sel
    [~,base,~] = fileparts(imgs(k).name);
    mp = '';
    for e = {'.png','.jpg','.jpeg','.tif','.bmp'}
        cand = fullfile(mskDir,[base e{1}]);
        if exist(cand,'file'), mp = cand; break; end
    end
    if isempty(mp), continue; end
    I = imread(fullfile(imgDir,imgs(k).name));
    if size(I,3)==1, I = cat(3,I,I,I); end
    I = im2uint8(imresize(I, inSize(1:2)));
    Mk = imread(mp); if size(Mk,3)>1, Mk = rgb2gray(Mk); end
    Mk = imresize(Mk, inSize(1:2), 'nearest');
    gt = double(Mk(:) > 127);            % 1 = forged
    [~,~,allScores] = semanticseg(I, net);   % HxWxNumClasses
    sF = allScores(:,:,forgedIdx); sF = double(sF(:));
    allScore = [allScore; sF];           %#ok<AGROW>
    allLabel = [allLabel; gt];           %#ok<AGROW>
    cnt = cnt + 1;
end
fprintf('Inference done on %d images, %d pixels.\n', cnt, numel(allLabel));

%% ROC + AUC (manual, threshold sweep)
thr = linspace(0,1,500);
P = sum(allLabel==1); Nn = sum(allLabel==0);
TPR = zeros(size(thr)); FPR = zeros(size(thr));
PRE = zeros(size(thr)); REC = zeros(size(thr));
for i=1:numel(thr)
    pred = allScore >= thr(i);
    TP = sum(pred & allLabel==1); FP = sum(pred & allLabel==0); FN = sum(~pred & allLabel==1);
    TPR(i) = TP/max(P,1); FPR(i) = FP/max(Nn,1);
    PRE(i) = TP/max(TP+FP,1); REC(i) = TPR(i);
end
[FPRs, idx] = sort(FPR); TPRs = TPR(idx);
AUC = trapz(FPRs, TPRs);
[RECs, idr] = sort(REC); PREs = PRE(idr);
AUC_PR = trapz(RECs, PREs);
fprintf('AUC(ROC)=%.4f  AUC(PR)=%.4f\n', AUC, AUC_PR);

% ROC figure
f2 = figure('Color','w','Visible','off','Position',[100 100 460 400]);
plot(FPRs, TPRs,'k-','LineWidth',2); hold on; grid on;
plot([0 1],[0 1],'--','Color',[.5 .5 .5]);
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title(sprintf('ROC Curve — Forged class (AUC = %.3f)', AUC));
legend({sprintf('U-Net (AUC=%.3f)',AUC),'Chance'},'Location','southeast');
axis([0 1 0 1]);
saveas(f2, fullfile(outDir,'roc_curve.png')); close(f2);

% PR / AUC figure
f3 = figure('Color','w','Visible','off','Position',[100 100 460 400]);
plot(RECs, PREs,'k-','LineWidth',2); grid on;
xlabel('Recall'); ylabel('Precision');
title(sprintf('Precision–Recall Curve — Forged class (AUC = %.3f)', AUC_PR));
axis([0 1 0 1]);
saveas(f3, fullfile(outDir,'auc_pr_curve.png')); close(f3);

save(fullfile(outDir,'roc_data.mat'),'FPRs','TPRs','PREs','RECs','AUC','AUC_PR','cm');
fprintf('All document figures saved to %s\n', outDir);
