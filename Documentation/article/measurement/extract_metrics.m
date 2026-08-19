%% extract_metrics.m - dump every stored result artefact to text for documentation
root = 'F:\Current_Work\SemanticSegmentationUsingFCN-AlexNet';
out  = 'C:\Users\USER\AppData\Local\Temp\claude\F--Current-Work-SemanticSegmentationUsingFCN-AlexNet\809c6f5e-39a4-4d1b-bc50-83811f54d99e\scratchpad';
variants = {'baseline','deeplab','improved','transfer'};

fid = fopen(fullfile(out,'metrics_dump.txt'),'w');

for v = 1:numel(variants)
    vn = variants{v};
    folder = fullfile(root, sprintf('Improved_Segmentation_Results_%s', vn));
    fprintf(fid, '===== VARIANT %s =====\n', vn);
    pm = fullfile(folder,'PixelMetrics_improved.mat');
    if exist(pm,'file')
        S = load(pm);
        fn = fieldnames(S);
        fprintf(fid, 'vars in PixelMetrics: %s\n', strjoin(fn', ', '));
        if isfield(S,'cm')
            fprintf(fid, 'CM (rows=true Bg,Fg; cols=pred Bg,Fg):\n');
            fprintf(fid, '  %d %d\n  %d %d\n', S.cm(1,1),S.cm(1,2),S.cm(2,1),S.cm(2,2));
            fprintf(fid, 'CM total pixels: %.0f\n', sum(S.cm(:)));
        end
        if isfield(S,'M')
            fprintf(fid, 'globalAcc = %.10f\n', S.M.globalAcc);
            fprintf(fid, 'precision = %s\n', mat2str(S.M.precision',10));
            fprintf(fid, 'recall    = %s\n', mat2str(S.M.recall',10));
            fprintf(fid, 'f1        = %s\n', mat2str(S.M.f1',10));
            fprintf(fid, 'IoU       = %s\n', mat2str(S.M.IoU',10));
        end
        if isfield(S,'bestArea'), fprintf(fid,'bestArea (minArea) = %d\n', S.bestArea); end
        if isfield(S,'trainSecs'), fprintf(fid,'trainSecs = %.2f (%.2f min, %.2f h)\n', S.trainSecs, S.trainSecs/60, S.trainSecs/3600); end
        if isfield(S,'sampleInfo')
            fprintf(fid,'sample names: %s\n', strjoin(S.sampleInfo.names, ', '));
        end
        if isfield(S,'cfg')
            c = S.cfg;
            f2 = fieldnames(c);
            for k = 1:numel(f2)
                val = c.(f2{k});
                if ischar(val) || isstring(val)
                    fprintf(fid,'cfg.%s = %s\n', f2{k}, char(strjoin(string(val),'|')));
                elseif isnumeric(val) || islogical(val)
                    fprintf(fid,'cfg.%s = %s\n', f2{k}, mat2str(double(val)));
                end
            end
        end
    else
        fprintf(fid, '(no PixelMetrics_improved.mat)\n');
    end

    ns = fullfile(folder,'netSeg_improved.mat');
    if exist(ns,'file')
        try
            W = whos('-file', ns);
            fprintf(fid,'netSeg file vars: ');
            for k=1:numel(W), fprintf(fid,'%s(%s) ', W(k).name, W(k).class); end
            fprintf(fid,'\n');
            Q = load(ns,'trIdx','vaIdx','teIdx');
            if isfield(Q,'trIdx')
                fprintf(fid,'split sizes: train %d | val %d | test %d\n', ...
                    numel(Q.trIdx), numel(Q.vaIdx), numel(Q.teIdx));
            end
            R = load(ns,'info');
            if isfield(R,'info')
                inf_ = R.info;
                if isstruct(inf_) || isobject(inf_)
                    try
                        th = inf_.TrainingHistory;
                        fprintf(fid,'TrainingHistory rows: %d; vars: %s\n', height(th), strjoin(th.Properties.VariableNames, ', '));
                        writetable(th, fullfile(out, sprintf('trainhist_%s.csv', vn)));
                    catch ME
                        fprintf(fid,'no TrainingHistory (%s)\n', ME.message);
                    end
                    try
                        vh = inf_.ValidationHistory;
                        fprintf(fid,'ValidationHistory rows: %d; vars: %s\n', height(vh), strjoin(vh.Properties.VariableNames, ', '));
                        writetable(vh, fullfile(out, sprintf('valhist_%s.csv', vn)));
                    catch ME
                        fprintf(fid,'no ValidationHistory (%s)\n', ME.message);
                    end
                    try
                        pe = inf_.PerEpoch;
                        if ~isempty(pe)
                            fprintf(fid,'PerEpoch rows: %d\n', height(pe));
                            writetable(pe, fullfile(out, sprintf('perepoch_%s.csv', vn)));
                        end
                    catch
                    end
                end
            end
        catch ME
            fprintf(fid,'netSeg load error: %s\n', ME.message);
        end
    end
    fprintf(fid,'\n');
end

% --- older tuned-model ROC data ---
rd = fullfile(root,'Documentation','figures','roc_data.mat');
if exist(rd,'file')
    S = load(rd);
    fprintf(fid,'===== roc_data.mat =====\nvars: %s\n', strjoin(fieldnames(S)', ', '));
    fnm = fieldnames(S);
    for k=1:numel(fnm)
        val = S.(fnm{k});
        fprintf(fid,'  %s: %s %s\n', fnm{k}, class(val), mat2str(size(val)));
        if isnumeric(val) && numel(val)==1
            fprintf(fid,'     value = %.10f\n', val);
        end
    end
end
fclose(fid);
disp('done');
