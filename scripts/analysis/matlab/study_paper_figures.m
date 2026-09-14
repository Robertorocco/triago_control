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
%% 6. Every remaining metric
% Sections 3 to 5 showed the headline quantities. Everything else the analysis
% measures is below, grouped by family and drawn and read exactly the same way.
% Nothing is cherry-picked: every metric that can be put on a six-condition
% chart is here.
%
% *One group cannot be drawn this way.* The haptic force metrics (mean, peak,
% impulse) and the clutch-button metrics are absent on purpose: force is
% rendered by a different law in each control mode, and the clutch button only
% exists in clutch mode. Putting either on a chart that compares clutch against
% joystick would compare the apparatus, not the operator. They are analysed
% _within_ each mode in the full report (|study_report|) — that is where to look
% for the effort question in the clutch column specifically.
%% 6a. Time and effectiveness — the rest
% *Time under human control* is the task time minus the seconds the robot spent
% performing the grasp by itself: the part the operator is actually responsible
% for. *Path efficiency* is straight-line distance divided by distance actually
% travelled — 1.0 would be a perfect straight line, lower means more wandering.
% *Mean hand speed* is how fast the active hand moved. *Time in autonomous grasp
% phases* is how long the robot drove itself; it is neither good nor bad, it
% just says how much of the trial was handed over to the machine.

fig_paper_panels(S, items, ["teleop_time_s", "ee_path_efficiency", ...
                 "ee_speed_mean_mps", "autonomy_grasp_time_s"], ...
                 'cols', 1, 'panel_h', 235, 'name', 'time and effectiveness (rest)');
%% 6b. Safety — the rest
% *Near-miss time fraction* is the share of the operator-driven trial spent
% closer than 5 cm to something, and *near-miss episodes* counts how many
% separate times it dipped below that line — one long approach and ten brief
% scares look very different to an operator but can average the same. *Mean
% barrier multiplier* is how hard the safety filter was pushing on average, not
% just whether it was on.

fig_paper_panels(S, items, ["safety_nearmiss_frac", "safety_nearmiss_episodes", ...
                 "cbf_lambda_mean"], 'cols', 1, 'panel_h', 235, 'name', 'safety (rest)');
%% 6c. Motion quality — the rest
% *Mean tracking slack* is how far the robot was allowed to fall behind the
% reference it was given: large slack means the controller was relaxing the
% tracking constraint to stay feasible, usually because it was busy avoiding
% something. *Peak commanded joint rate* is the single fastest joint command in
% the trial — the spikes that a mean hides.

fig_paper_panels(S, items, ["slack_mean", "qdot_cmd_max"], ...
                 'cols', 1, 'panel_h', 250, 'name', 'motion quality (rest)');
%% 6d. Did the robot understand what the operator wanted?
% These describe the intent estimate: the robot continuously guesses which
% object the operator is reaching for, and these say how well that went.
%
% *Peak intent confidence* is the highest probability it ever assigned to a
% single goal (1.0 = certain). *Time to confident intent* is how many seconds
% until it first passed 80% confidence — lower is better, and it is blank for
% trials where it never got there. *Intent ever confident* is the fraction of
% trials in which it passed that line at all, so 1.0 means "always worked out
% eventually" and 0.5 means "half the time it never became sure".

fig_paper_panels(S, items, ["belief_max_prob", "belief_time_to_conf_s", ...
                 "belief_confident_ever"], 'cols', 1, 'panel_h', 235, 'name', 'intent understanding');
%% 6e. How well did operator and robot agree? (blending conditions only)
% These four exist only where the robot is actually blending its motion into the
% command — the *B* and *FB* conditions. The *F* conditions have no blending to
% measure, so these charts have *four bars, not six*.
%
% *User-autonomy agreement* is how closely the direction the operator was
% pushing matched the direction the robot wanted to go (1.0 = same direction,
% 0 = perpendicular): high means the help was pulling with the operator rather
% than against them. *Mean autonomy authority* is how much of the motion the
% robot contributed on average (0 = all operator, 1 = all robot), and
% *autonomy-led time* is the share of the trial where it contributed more than
% half. *User actively driving* is how much of the trial the operator was
% actually commanding something rather than holding still.

fig_paper_panels(S, items, ["agreement_mean_cos", "alpha_mean", ...
                 "alpha_autonomy_frac", "user_active_frac"], ...
                 'cols', 1, 'panel_h', 235, 'name', 'assistance quality');
%% 6f. The remaining family scores
% The same z-scored, higher-is-better summaries as section 5, for the three
% families not shown there. *Human effort* here rests on the joint-rate demand
% only, for the cross-mode reason given at the top of this section.

fig_paper_panels(S, items, ["fam_human_effort", "fam_intent_understanding", ...
                 "fam_assistance_quality"], 'cols', 1, 'panel_h', 235, 'name', 'remaining family scores');
%% 7. What this page does not tell you
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
