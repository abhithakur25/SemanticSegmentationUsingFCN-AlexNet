%% Textured Column Bar Graph — 400x300 px with Adaptive In-Plot Vertical Legend
% Author: Dr. Abhishek Thakur (final version)
% Description:
%   Grouped textured column bars with spacing and adaptive legend placement.
%   - Legend is vertically oriented inside the plot (stacked down)
%   - No title
%   - No X-label rotation
%   - Vertical axis labeled as "Accuracy"
%   - All legend textures displayed neatly inside graph.
%
% Image output: 400x300 pixels, self-contained (no external functions).

clear; close all; clc;

% -------------------------
% USER INPUT
% -------------------------
fname = 'D:\claude\SemanticSegmentationUsingFCN-AlexNet1\Final_Segmentation_Results_Tuned\PerClass_PixelMetrics.csv';  % First column = labels, rest = data

% -------------------------
% LOAD DATA
% -------------------------
T = readtable(fname);
labels = string(T{:,1});
Y = table2array(T(:,2:end));
[nGroups, nSeries] = size(Y);

% -------------------------
% FIGURE SETUP
% -------------------------
fig = figure('Color','w', 'Units','pixels', 'Position',[100 100 400 300], ...
             'PaperPositionMode','auto', 'Renderer','opengl');
ax = axes('Parent',fig, 'Box','on');
hold(ax,'on');

% -------------------------
% LAYOUT SETTINGS
% -------------------------
groupSpacing = 0.45;
groupWidth = 0.65;
barWidth = groupWidth / max(1,nSeries);
groupCenters = (1:nGroups) * (1 + groupSpacing);
xTicks = groupCenters;

% -------------------------
% TEXTURE PATTERNS
% -------------------------
textureNames = {'slash','backslash','vertical','horizontal','plus','cross',...
                'dots_sparse','checker','diagonal_grid','wave','thin_hatch','thick_hatch'};
tileSz = 120;
patternImgs = cell(1,nSeries);
for s = 1:nSeries
    tname = textureNames{mod(s-1,numel(textureNames))+1};
    patternImgs{s} = generateTextureByName(tname, tileSz);
end

% -------------------------
% DRAW BARS
% -------------------------
for s = 1:nSeries
    for g = 1:nGroups
        xpos = groupCenters(g) - groupWidth/2 + (s-1)*barWidth;
        w = barWidth * 0.88;
        val = Y(g,s);
        if isnan(val) || val == 0, continue; end

        % Border rectangle
        rectangle('Position',[xpos, 0, w, val], 'EdgeColor','k', 'LineWidth',0.6);

        % Texture fill
        img = flipud(patternImgs{s});
        X = [xpos, xpos+w; xpos, xpos+w];
        Ymat = [0, 0; val, val];
        surf(X, Ymat, zeros(2,2), 'CData', img, 'FaceColor', 'texturemap', ...
             'EdgeColor','none', 'FaceLighting','none');
    end
end

% -------------------------
% AXIS FORMAT
% -------------------------
xlim([min(groupCenters)-0.3, max(groupCenters)+0.5]);
ylim([0, max(Y(:))*1.18]);
set(ax, 'XTick', xTicks, 'XTickLabel', labels, 'FontName','Times New Roman', 'FontSize', 9);
ylabel('Accuracy','FontName','Times New Roman','FontSize',10);
grid(ax,'on');
set(ax,'Layer','top');

% -------------------------
% VERTICAL LEGEND INSIDE PLOT (Right Side)
% -------------------------
axXLim = xlim(ax); axYLim = ylim(ax);
xRange = axXLim(2) - axXLim(1);
yRange = axYLim(2) - axYLim(1);

% Place legend vertically along right blank area
legendX = axXLim(2) - xRange * 0.22;   % slightly inside from right border
legendYTop = axYLim(2) - yRange * 0.05; % start a bit below top
sampleW = xRange * 0.05;                % width of texture sample
sampleH = yRange * 0.05;                % height of each legend sample
gapY = sampleH * 1.2;                   % vertical spacing between samples

% Adjust if legend exceeds height
if (nSeries * gapY) > yRange * 0.9
    sampleH = (yRange * 0.9) / nSeries;
    gapY = sampleH * 1.1;
end

% Draw each legend item vertically
for s = 1:nSeries
    yB = legendYTop - (s-1)*gapY - sampleH;
    xL = legendX;
    img = flipud(patternImgs{s});
    X = [xL, xL+sampleW; xL, xL+sampleW];
    Ymat = [yB, yB; yB+sampleH, yB+sampleH];
    surf(X, Ymat, zeros(2,2), 'CData', img, 'FaceColor','texturemap', 'EdgeColor','k');

    % Label aligned to the right of the box
    text(xL + sampleW + 0.01*xRange, yB + sampleH/2, ...
         strrep(T.Properties.VariableNames{s+1},'_',' '), ...
         'VerticalAlignment','middle', 'FontName','Times New Roman', 'FontSize', 7);
end

% Border around legend column
rectangle('Position',[legendX - 0.01*xRange, legendYTop - nSeries*gapY - 0.02*yRange, ...
          sampleW + 0.15*xRange, nSeries*gapY + 0.03*yRange], ...
          'EdgeColor',[0.3 0.3 0.3], 'LineStyle','--', 'LineWidth',0.5);

hold(ax,'off');

% -------------------------
% Helper: Generate Texture Patterns
% -------------------------
function img = generateTextureByName(name, sz)
    switch lower(name)
        case 'slash',          img = gen_slash(sz, round(sz/12), max(1,round(sz/200)));
        case 'backslash',      img = gen_backslash(sz, round(sz/12), max(1,round(sz/200)));
        case 'vertical',       img = gen_vertical(sz, round(sz/12), max(1,round(sz/200)));
        case 'horizontal',     img = gen_horizontal(sz, round(sz/12), max(1,round(sz/200)));
        case 'plus',           img = gen_plus(sz, round(sz/14), max(1,round(sz/220)));
        case 'cross',          img = gen_cross(sz, round(sz/14), max(1,round(sz/220)));
        case 'dots_sparse',    img = gen_dots(sz, round(sz/10), max(1,round(sz/60)));
        case 'checker',        img = gen_checker(sz, round(sz/12));
        case 'diagonal_grid',  img = gen_diagonal_grid(sz, round(sz/14), max(1,round(sz/220)));
        case 'wave',           img = gen_wave(sz, round(sz/14));
        case 'thin_hatch',     img = gen_thin_hatch(sz, round(sz/18));
        case 'thick_hatch',    img = gen_thick_hatch(sz, round(sz/8));
        otherwise,             img = ones(sz,sz,3);
    end
end

% -------------------------
% Texture Pattern Generators
% -------------------------
function out = gen_slash(sz, spacing, t)
    mask = ones(sz);
    for k = -2*sz:spacing:2*sz
        x = 1:sz; y = x + k; valid = y>=1 & y<=sz;
        for ii = find(valid), yy = round(y(ii)); mask(max(1,yy-t):min(sz,yy+t), ii) = 0; end
    end, out = mask2rgb(mask);
end

function out = gen_backslash(sz, spacing, t)
    mask = ones(sz);
    for k = -2*sz:spacing:2*sz
        x = 1:sz; y = -x + (sz + k); valid = y>=1 & y<=sz;
        for ii = find(valid), yy = round(y(ii)); mask(max(1,yy-t):min(sz,yy+t), ii) = 0; end
    end, out = mask2rgb(mask);
end

function out = gen_vertical(sz, spacing, t)
    mask = ones(sz);
    for c = 1:spacing:sz, mask(:, max(1,c-t):min(sz,c+t)) = 0; end, out = mask2rgb(mask);
end

function out = gen_horizontal(sz, spacing, t)
    mask = ones(sz);
    for r = 1:spacing:sz, mask(max(1,r-t):min(sz,r+t), :) = 0; end, out = mask2rgb(mask);
end

function out = gen_plus(sz, spacing, t)
    mask = ones(sz);
    for c = 1:spacing:sz, mask(:, max(1,c-t):min(sz,c+t)) = 0; end
    for r = 1:spacing:sz, mask(max(1,r-t):min(sz,r+t), :) = 0; end, out = mask2rgb(mask);
end

function out = gen_cross(sz, spacing, t)
    mask = ones(sz);
    for k = -2*sz:spacing:2*sz, x = 1:sz; y = x + k; valid = y>=1 & y<=sz;
        for ii = find(valid), yy = round(y(ii)); mask(max(1,yy-t):min(sz,yy+t), ii) = 0; end
    end
    for k = -2*sz:spacing:2*sz, x = 1:sz; y = -x + (sz + k); valid = y>=1 & y<=sz;
        for ii = find(valid), yy = round(y(ii)); mask(max(1,yy-t):min(sz,yy+t), ii) = 0; end
    end, out = mask2rgb(mask);
end

function out = gen_dots(sz, spacing, radius)
    mask = ones(sz);
    for r = round(spacing/2):spacing:sz
        for c = round(spacing/2):spacing:sz
            [rr,cc] = ndgrid(1:sz,1:sz);
            mask(((rr-r).^2 + (cc-c).^2) <= radius^2) = 0;
        end
    end, out = mask2rgb(mask);
end

function out = gen_checker(sz, block)
    mask = ones(sz);
    for r = 1:block:sz
        for c = 1:block:sz
            if mod(floor((r-1)/block) + floor((c-1)/block),2) == 0
                mask(r:min(sz,r+block-1), c:min(sz,c+block-1)) = 0;
            end
        end
    end, out = mask2rgb(mask);
end

function out = gen_diagonal_grid(sz, spacing, t)
    mask = ones(sz);
    for k = -2*sz:spacing:2*sz, x = 1:sz; y = x + k; valid = y>=1 & y<=sz;
        for ii = find(valid), yy = round(y(ii)); mask(max(1,yy-t):min(sz,yy+t), ii) = 0; end
    end
    for k = -2*sz:spacing:2*sz, x = 1:sz; y = -x + (sz + k); valid = y>=1 & y<=sz;
        for ii = find(valid), yy = round(y(ii)); mask(max(1,yy-t):min(sz,yy+t), ii) = 0; end
    end, out = mask2rgb(mask);
end

function out = gen_wave(sz, spacing)
    mask = ones(sz); amp = max(1,round(sz/28)); freq = 2*pi/(spacing*2);
    for c = 1:sz, mid = round((sz/2) + amp * sin(c*freq)); mask(max(1,mid-1):min(sz,mid+1), c) = 0; end
    out = mask2rgb(mask);
end

function out = gen_thin_hatch(sz, spacing)
    mask = ones(sz); t = 1; for c = 1:spacing:sz, mask(:, max(1,c-t):min(sz,c+t)) = 0; end
    out = mask2rgb(mask);
end

function out = gen_thick_hatch(sz, spacing)
    mask = ones(sz); t = max(2, round(sz/80)); for c = 1:spacing:sz, mask(:, max(1,c-t):min(sz,c+t)) = 0; end
    out = mask2rgb(mask);
end

function rgb = mask2rgb(mask)
    if ~isfloat(mask), mask = double(mask); end
    rgb = repmat(mask,1,1,3);
end
