function fig = fig_success_incidents(trial)
%FIG_SUCCESS_INCIDENTS Success rate and incident rate per cell (descriptive).
%   FIG = FIG_SUCCESS_INCIDENTS(TRIAL) shows, for each of the six cells, the
%   percentage of trials marked successful by the experimenter and the
%   percentage with a note (fallen object, failed grasp, ...). These rates
%   are near the ceiling / floor, so they are reported as counts rather
%   than tested.

cells = ["CF" "CB" "CFB" "JF" "JB" "JFB"];
nC = numel(cells);
succ = nan(1, nC); inc = nan(1, nC); ntr = nan(1, nC);
for j = 1:nC
    sel = trial.cell == cells(j);
    ntr(j) = nnz(sel);
    succ(j) = 100 * mean(trial.success(sel));
    inc(j) = 100 * mean(trial.incident(sel));
end
fig = studyplot.newfig("Success and incidents", 1000, 420);
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl, sprintf("Task success and incidents per cell  (%d trials, %d participants)", height(trial), numel(unique(trial.participant))), ...
      'FontWeight', 'bold', 'FontSize', 13);

ax = nexttile(tl);
studyplot.bars(ax, succ, nan(1, nC), nan(1, nC), cells, 'ylabel', "% of trials", 'labels', cells);
ylim(ax, [0 105]);
for j = 1:nC, text(ax, j, succ(j) + 2, sprintf("%d/%d", round(succ(j) / 100 * ntr(j)), ntr(j)), 'HorizontalAlignment', 'center', 'FontSize', 9); end
title(ax, "Success rate (experimenter's call) - higher is better");

ax = nexttile(tl);
studyplot.bars(ax, inc, nan(1, nC), nan(1, nC), cells, 'ylabel', "% of trials", 'labels', cells);
ylim(ax, [0 max(30, max(inc) + 10)]);
for j = 1:nC, text(ax, j, inc(j) + 1, sprintf("%d/%d", round(inc(j) / 100 * ntr(j)), ntr(j)), 'HorizontalAlignment', 'center', 'FontSize', 9); end
title(ax, "Incident noted (fallen object, failed grasp, ...) - lower is better");
end
