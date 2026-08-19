%% fix_roc.m - recompute ROC/PR from the saved score histograms.
% roc_transfer.m swept the threshold from low to high, so FPR/TPR came out in
% DECREASING order; prepending (0,0) to the front of that sequence made trapz
% integrate backwards over a spurious first segment. Here the operating points
% are re-ordered by increasing FPR and the endpoints (0,0) and (1,1) appended
% at the correct ends. No inference is repeated - only the histograms are used.

out = 'C:\Users\USER\AppData\Local\Temp\claude\F--Current-Work-SemanticSegmentationUsingFCN-AlexNet\809c6f5e-39a4-4d1b-bc50-83811f54d99e\scratchpad';
S = load(fullfile(out,'roc_transfer.mat'),'hPos','hNeg','summary');
hPos = S.hPos(:); hNeg = S.hNeg(:);
NB = numel(hPos);

P = sum(hPos); N = sum(hNeg);

% bin k and above predicted Forged  =>  index 1 is "predict everything Forged"
TP = flipud(cumsum(flipud(hPos)));
FP = flipud(cumsum(flipud(hNeg)));
FN = P - TP;
TN = N - FP;

TPR = TP / P;
FPR = FP / N;
PRE = TP ./ max(TP + FP, 1);
REC = TPR;
F1  = 2*TP ./ max(2*TP + FP + FN, 1);
MCC = (TP.*TN - FP.*FN) ./ max(sqrt((TP+FP).*(TP+FN).*(TN+FP).*(TN+FN)), 1);
thr = ((0:NB-1)' + 0.5) / NB;

% --- ROC: order by increasing FPR, close the curve at both ends ---
x = [0; flipud(FPR); 1];
y = [0; flipud(TPR); 1];
[x, iu] = unique(x, 'stable');
y = y(iu);
[x, is] = sort(x); y = y(is);
AUC = trapz(x, y);

% --- PR: order by increasing recall; anchor at recall 0 with the precision
%     of the most confident bin (standard convention), and at recall 1. ---
r = flipud(REC); p = flipud(PRE);
keep = r > 0;                     % drop degenerate zero-recall points
r = [0; r(keep)];
p = [p(find(keep,1)); p(keep)];
[r, iu] = unique(r, 'stable'); p = p(iu);
[r, is] = sort(r); p = p(is);
AUC_PR = trapz(r, p);

[bestF1, kF1]   = max(F1);
[bestMCC, kM]   = max(MCC);
half = round(0.5*NB);

fprintf('\n=== corrected threshold-free metrics ===\n');
fprintf('pixels scored     = %.0f  (positives %.0f, negatives %.0f)\n', P+N, P, N);
fprintf('AUC(ROC)          = %.6f\n', AUC);
fprintf('AUC(PR)           = %.6f\n', AUC_PR);
fprintf('Best F1           = %.6f at threshold %.4f\n', bestF1, thr(kF1));
fprintf('Best MCC          = %.6f at threshold %.4f\n', bestMCC, thr(kM));
fprintf('MCC   @0.5        = %.6f\n', MCC(half));
fprintf('F1    @0.5        = %.6f\n', F1(half));
fprintf('Prec  @0.5        = %.6f\n', PRE(half));
fprintf('Sens  @0.5        = %.6f\n', TPR(half));
fprintf('Spec  @0.5        = %.6f\n', TN(half)/N);
fprintf('prevalence        = %.6f\n', P/(P+N));

% Plot-ready curves: sample densely where the curve bends.
sel = unique([1:50, round(logspace(log10(51), log10(NB), 900)), NB]);
Tbl = table(thr(sel), FPR(sel), TPR(sel), PRE(sel), REC(sel), F1(sel), MCC(sel), ...
    'VariableNames', {'Threshold','FPR','TPR','Precision','Recall','F1','MCC'});
writetable(Tbl, fullfile(out,'roc_transfer_curve.csv'));

fid = fopen(fullfile(out,'roc_transfer_summary.txt'),'w');
fprintf(fid,'N_images = %d\n', S.summary.N_images);
fprintf(fid,'N_pixels = %.0f\n', P+N);
fprintf(fid,'AUC = %.8f\n', AUC);
fprintf(fid,'AUC_PR = %.8f\n', AUC_PR);
fprintf(fid,'bestF1 = %.8f\n', bestF1);
fprintf(fid,'bestF1_thr = %.6f\n', thr(kF1));
fprintf(fid,'bestMCC = %.8f\n', bestMCC);
fprintf(fid,'bestMCC_thr = %.6f\n', thr(kM));
fprintf(fid,'MCC_at_half = %.8f\n', MCC(half));
fprintf(fid,'F1_at_half = %.8f\n', F1(half));
fprintf(fid,'precision_at_half = %.8f\n', PRE(half));
fprintf(fid,'sensitivity = %.8f\n', TPR(half));
fprintf(fid,'specificity = %.8f\n', TN(half)/N);
fprintf(fid,'prevalence = %.8f\n', P/(P+N));
fclose(fid);
disp('roc summary rewritten.');
