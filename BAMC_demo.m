clc;
close all;

run(fullfile('funs', 'setup_path.m'));

load('UCI');
c = max(Y);

As = cellfun(@(Xv) constructW_PKN(Xv, 5), X, 'uni', 0);

gamma = 5;
r = 1.5;  % 0 < r < 2
y_pred = BAMC(As, c, gamma, r);

result = ClusteringMeasure_new(Y, y_pred)
