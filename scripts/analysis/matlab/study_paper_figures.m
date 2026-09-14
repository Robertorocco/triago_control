%% TRIAGo user study — results at a glance
% This document is the *short, visual* summary of the study: a handful of
% figures in the layout a robotics paper uses for its results section, each one
% explained in plain words. It is meant to be *read*, not run — everything you
% need is on this page.
%
% For the full statistical report (every metric, every test, every assumption
% check) open |study_report| instead. This document deliberately shows only the
% headline quantities.
%
% *Status.* Recording is still in progress. Every number here is computed from
% the participants finished so far and will move as more are added; the figures
% redraw themselves from whatever is in the export at the time they are built.

%%
warning('off', 'MATLAB:hg:AutoSoftwareOpenGL');   % headless render notice, not a result
RESULTS_DIR = '';   % '' = most recent run (analysis_results/latest.txt)
EXPORT_DIR  = '';   % '' = auto-locate manifest.mat (see study_export_dir)

% The six headline quantities, one per panel of the main figure. Any metric of
% study_metric_spec that is defined in all six cells can go here.
METRICS = ["duration_s", "ee_path_len_m", "safety_min_dist_m", ...
           "cbf_active_frac", "ee_sparc", "qdot_cmd_rms"];

% The summary panels: the overall score and the family scores behind it.
SUMMARY_METRICS = ["composite", "fam_time_effectiveness", "fam_safety", "fam_motion_quality"];

if isempty(RESULTS_DIR)
    root = fullfile(study_export_dir(EXPORT_DIR), 'analysis_results');
    latest = fullfile(root, 'latest.txt');
    if ~isfile(latest)
        error('study_paper_figures:noResults', ...
              ['No analysis results found.\nRun  run_study_analysis  first ' ...
               '(it writes %s).'], latest);
    end
    RESULTS_DIR = strtrim(fileread(latest));
end
load(fullfile(RESULTS_DIR, 'results.mat'), 'trial', 'items', 'meta', 'S', 'settings');
alpha = settings.alpha;
%% 1. What the study compares
% Every participant drove the TRIAGo robot through the same pick-and-place task
% with a Haption force-feedback handle, under *six different ways of being
% helped*. The six conditions are every combination of two things:
%
% * *How the hand is mapped to the robot* — two levels:
% *C = Clutch* (the handle's position becomes the robot's position; a button
% re-centres the handle when you run out of room) and *J = Joystick* (the
% handle's offset from a spring-centred rest pose becomes a velocity, so the
% robot keeps moving while the handle is held away from centre).
% * *What kind of help the robot gives* — three levels:
% *F = Feedback* (the handle pulls you toward the goal the robot thinks you
% want), *B = Blending* (the robot quietly mixes its own motion toward that
% goal into your command), and *FB = both at once*.
%
% That gives the six conditions used in every figure below, always in this
% order: *CF, CB, CFB* (clutch) then *JF, JB, JFB* (joystick).
%
% Each person did all six conditions, in two different scenes, so *every
% comparison in this document is made within the same person* — condition
% against condition inside one operator. That removes the biggest source of
% noise in this kind of study, which is that people simply differ from each
% other.

fprintf('Participants analysed : %d complete\n', meta.n_complete);
fprintf('%s\n', studyplot.wrap(strjoin(meta.complete_list, ', '), 66));
fprintf('Trials analysed       : %d\n', height(trial));
if ~isempty(meta.excluded)
    fprintf('\nNot yet included (recording incomplete):\n');
    disp(meta.excluded)
end
%% 2. How to read every figure on this page
% All the figures share one layout, so learning it once is enough:
%
% * *Each bar is one of the six conditions*, always in the same order, with
% clutch conditions in blue and joystick conditions in orange. Within a colour,
% the shade gets darker as more help is given (F → B → FB).
% * *The height of the bar is the average across participants* — one number per
% person first, then averaged, so nobody counts twice.
% * *The thin vertical whisker is the standard error*: roughly how much the
% average itself would wobble if the study were repeated with a different group
% of people. Short whisker = the average is well pinned down.
% * *A horizontal bracket with a p-value on top* marks a pair of conditions
% that really did differ. Pairs without a bracket were compared too — they just
% did not come out different.
% * Under each panel title it says whether *higher or lower is better* for that
% quantity, and how many participants it is based on.
%
% *What the p-value means.* It answers one question only: _if these two
% conditions were truly identical, how often would chance alone produce a gap
% this big?_ A p of 0.01 means "about 1 time in 100" — unlikely enough that we
% call the difference real. The usual cut-off is 0.05. Because we compare many
% pairs at once, and testing many pairs makes lucky-looking gaps more likely,
% every p-value shown has been *corrected for that* (Holm correction): they are
% already the conservative numbers.
%
% *One thing to watch.* A couple of panels say _"clipped axis"_
% under the title. There the bars do not start at zero, because the values sit
% in a narrow band far from it and drawing from zero would make all six bars
% look identical. On those panels judge the differences by the numbers and the
% brackets, not by how many times taller one bar looks than another.
%% 3a. Was it quicker and more direct?
% *Task time* is simply how long the trial took. *Hand path length* is the total
% distance the two grippers travelled to do it: for the same task, a shorter
% path means less wandering, fewer corrections and less backtracking. Lower is
% better in both panels.

fig_paper_panels(S, items, METRICS(1:2), 'cols', 1, 'name', 'speed and directness');
%% 3b. Was it safer?
% *Minimum clearance* is the closest the robot ever came to an obstacle while
% the operator was driving — the autonomous grasp moments are excluded, because
% the gripper is _supposed_ to touch the object then. It is a signed margin
% measured from the safety envelope rather than from the metal, so the values
% are negative: the envelope is deliberately generous and gets entered often.
% Higher (less negative) is better.
%
% *Safety filter active* is the fraction of the trial in which the robot's
% collision barrier actually had to push back against the command it was given.
% This is the honest measure of "how often did the operator need saving" —
% lower means they stayed out of trouble by themselves.

fig_paper_panels(S, items, METRICS(3:4), 'cols', 1, 'name', 'safety');
%% 3c. Was the resulting motion better?
% *Smoothness (SPARC)* is a standard smoothness score for the hand's speed
% profile. It is always negative: *closer to zero means smoother*, more negative
% means the motion was jerky, hesitant or full of stop-and-go.
%
% *Commanded joint-rate RMS* is how vigorously the controller had to move the
% arm joints to follow the operator. Lower means calmer, less frantic motion.

fig_paper_panels(S, items, METRICS(5:6), 'cols', 1, 'name', 'motion quality');
%% 4. What the statistics say
% The figure above shows *where* the differences are. This section says which of
% them the statistics actually back, and how big they are.
%
% Two questions are asked of every quantity, using a two-way repeated-measures
% ANOVA — the standard test when the same people do every condition:
%
% * *Does the control mode matter?* (clutch vs joystick, averaged over the three
% kinds of help)
% * *Does the kind of help matter?* (F vs B vs FB, averaged over the two modes)
% * *Do the two interact?* — i.e. does the best kind of help _depend on_ which
% control mode is used. A significant interaction is the interesting case: it
% means there is no single best assistance, it depends on the mode.
%
% |partial_eta2| is the effect size: the share of the variation explained by
% that factor. Roughly, 0.01 is small, 0.06 medium, 0.14 large — it answers "is
% this difference big enough to care about", which the p-value alone cannot.

[E, Pairs] = paper_stats_tables(S, items, METRICS, alpha);
disp(E)
%%
% And the individual pairs of conditions that survived the correction. An empty
% table here is a perfectly normal result at this sample size: it means the
% overall pattern may be visible, but no single pair of the six is yet far
% enough apart to call on its own.

if isempty(Pairs)
    disp("No pair of conditions differs significantly after correction (yet).")
else
    disp(Pairs)
end
%% 5. The overall picture
% The last figure collapses everything into scores. Each metric is first put on
% a common scale (how far above or below that participant's own average it is),
% flipped so that *higher is always better*, and then averaged — first within a
% family of related metrics, then across families into one *composite score*.
%
% A score of 0 is the average condition. Positive is better than average,
% negative is worse. The unit is "standard deviations of this study", so a
% difference of 0.5 between two bars means half a standard deviation — a
% moderate, visible difference.
%
% This is the single most compact answer to "which condition worked best
% overall", but it is also the most summarised one: a condition can win on the
% composite while losing on a metric that matters more to you. Use it to get the
% shape of the result, then look back at panel 3 for the specifics.

fig_paper_panels(S, items, SUMMARY_METRICS, 'cols', 1, 'panel_h', 240, 'name', 'summary scores');
%% 6. What this page does not tell you
% * *The study is not finished.* Conditions that look close today can separate
% once more participants are in, and pairs that look significant can lose that
% status. Nothing here is final.
% * *Only objective measures are shown.* What the operators _felt_ — workload,
% trust, which condition they preferred — is collected through the post-trial
% questionnaire and is not part of this document yet. In the paper this study
% follows, that subjective panel often disagrees with the objective ones, so it
% is worth adding before drawing conclusions.
% * *Two quantities are deliberately missing*: the haptic force and the clutch
% button. Force is rendered differently in the two control modes and the clutch
% button only exists in clutch mode, so putting either on a chart that compares
% clutch against joystick would compare the apparatus, not the operator. They
% are analysed *within* each mode in the full report.
% * *"Not significant" is not the same as "no difference"* — with this many
% participants, only fairly large effects can be detected at all.

fprintf('\nBuilt from %s\n', RESULTS_DIR);
