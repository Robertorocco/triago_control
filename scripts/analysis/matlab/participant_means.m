function [M, participants] = participant_means(trial, metric, groupvar, levels, rowmask)
%PARTICIPANT_MEANS Participants x levels matrix of per-participant means of one metric.
%   [M, P] = PARTICIPANT_MEANS(TRIAL, METRIC, GROUPVAR, LEVELS) averages
%   TRIAL.(METRIC) over every trial of participant P(i) whose TRIAL.(GROUPVAR)
%   equals LEVELS(j), ignoring NaN (metrics outside their cell scope). A cell
%   with no defined trial stays NaN. ROWMASK (optional logical) restricts the
%   trials used, e.g. to one control mode.

if nargin < 5 || isempty(rowmask), rowmask = true(height(trial), 1); end
participants = unique(trial.participant, 'stable');
M = nan(numel(participants), numel(levels));
g = string(trial.(groupvar));
v = trial.(metric);
for i = 1:numel(participants)
    for j = 1:numel(levels)
        sel = rowmask & trial.participant == participants(i) & g == levels(j);
        if any(sel), M(i, j) = mean(v(sel), 'omitnan'); end
    end
end
end
