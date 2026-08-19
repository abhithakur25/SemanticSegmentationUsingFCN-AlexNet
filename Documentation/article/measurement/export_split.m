%% export_split.m - write the saved train/val/test index vectors to CSV so the
% Python duplication audit can use the exact partition the model was trained on.
out = 'C:\Users\USER\AppData\Local\Temp\claude\F--Current-Work-SemanticSegmentationUsingFCN-AlexNet\809c6f5e-39a4-4d1b-bc50-83811f54d99e\scratchpad';
S = load('F:\Current_Work\SemanticSegmentationUsingFCN-AlexNet\Improved_Segmentation_Results_transfer\netSeg_improved.mat', ...
         'trIdx','vaIdx','teIdx');
n = max([numel(S.trIdx) numel(S.vaIdx) numel(S.teIdx)]);
pad = @(v) [v(:); nan(n-numel(v),1)];
T = table(pad(double(S.trIdx)), pad(double(S.vaIdx)), pad(double(S.teIdx)), ...
    'VariableNames', {'trIdx','vaIdx','teIdx'});
writetable(T, fullfile(out,'split_idx.csv'), 'WriteVariableNames', true);
fprintf('exported: train %d, val %d, test %d\n', numel(S.trIdx), numel(S.vaIdx), numel(S.teIdx));
