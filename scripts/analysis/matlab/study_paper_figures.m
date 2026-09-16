%% TRIAGo user study — results at a glance
% This document is the *short, visual* summary of the study: a handful of
% figures in the layout a robotics paper uses for its results section, each one
% explained in plain words and followed by what the data actually say. It is
% meant to be *read*, not run — everything you need is on this page.
%
% For the full statistical report (every metric, every test, every assumption
% check) open |study_report| instead. This document deliberately shows only the
% headline quantities and the reading of them.
%
% *Status.* Data collection is complete: 24 participants, each doing all six
% conditions in both scenes (288 trials). The pilot session (P00) is not part
% of the analysis. The figures and tables below are recomputed from the export
% every time this document is built; the written commentary was made against
% that final data set.

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

% Everything else that is drawn in section 6, so the p-value census of
% section 2 counts exactly the comparisons on this page.
REST_METRICS = ["teleop_time_s", "ee_path_efficiency", "ee_speed_mean_mps", "autonomy_grasp_time_s", ...
                "safety_mean_dist_m", "safety_nearmiss_frac", "safety_nearmiss_s", "safety_nearmiss_episodes", ...
                "cbf_active_s", "cbf_lambda_active_median", ...
                "slack_mean", "qdot_cmd_max", ...
                "belief_mean_prob", "belief_max_prob", "belief_time_to_conf_s", ...
                "agreement_mean_cos", "alpha_mean", "alpha_autonomy_frac", "user_active_frac", ...
                "fam_human_effort", "fam_intent_understanding", "fam_assistance_quality"];

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
% items holds only the family metrics; the diagnostic views drawn in section 6
% (seconds next to a fraction) take their labels from the full catalogue.
full = study_metric_spec();
extra = full(~ismember(full.name, items.name), items.Properties.VariableNames(1:8));
extra.is_family = false(height(extra), 1);
items = [items; extra];
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
% other. Half the participants met the clutch conditions first and half the
% joystick conditions first, to balance out practice.

fprintf('Participants analysed : %d\n', meta.n_complete);
fprintf('%s\n', studyplot.wrap(strjoin(meta.complete_list, ', '), 66));
fprintf('Trials analysed       : %d\n', height(trial));
if ~isempty(meta.excluded)
    fprintf('\nExcluded from the analysis:\n');
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
% did not come out different. When there are more significant pairs than room,
% only the strongest are drawn; the table in section 4 lists them all.
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
% *Why so many brackets say p < 0.001 — and whether that is a problem.* It is
% not; it is what a clean within-person design looks like when the effect is
% real. Three things to know:
%
% # *A p-value measures certainty, not size.* p < 0.001 says "we are very sure
% these two differ"; it says nothing about _by how much_. A tiny difference
% that every participant shows will get p < 0.001 just as a huge one will. The
% size is the bar height difference (and, in section 4, |partial_eta2|). One
% panel below (peak intent confidence, 6d) has p < 0.001 everywhere for a
% difference of 0.003 — real, and irrelevant.
% # *p < 0.001 is a floor, not a ranking.* With 24 participants who all move
% the same way, the pairwise test simply bottoms out: it cannot report anything
% smaller than about 0.0003 after correction. Many of the brackets below carry
% that identical saturated value. Do not read "p < 0.001" as "stronger than
% p = 0.002" — both mean "as sure as this test can be".
% # *Not everything is significant, and the null results make physical sense.*
% Minimum clearance, how hard the safety filter pushes when it does push, time
% to a confident intent estimate, how much of the trial the operator was
% actively driving — none differ between conditions, and in each case there is
% a reason they should not. A study where _everything_ came
% out p < 0.001 would deserve suspicion; this one has its nulls where they
% belong.
%
% The count below is over every pairwise comparison drawn in this document. The
% "near-unanimous" pairs are the ones where practically every participant's
% ranking pointed the same way — that, not the p-value, is the strongest
% statement this study can make.

tot = 0; sig = 0; floor_ = 0; unanimous = 0;
for nm = [METRICS, SUMMARY_METRICS, REST_METRICS]
    if ~isfield(S.Q3, nm), continue; end
    P = S.Q3.(nm).pairs;
    if isempty(P) || ~ismember('significant', P.Properties.VariableNames), continue; end
    tot = tot + height(P);                 sig = sig + sum(P.significant);
    floor_ = floor_ + sum(P.p_holm < 0.001); unanimous = unanimous + sum(abs(P.r_rb) >= 0.9);
end
fprintf('Pairwise comparisons made          : %d\n', tot);
fprintf('  significant after Holm correction: %d\n', sig);
fprintf('  reported as p < 0.001            : %d   (the saturated floor, see above)\n', floor_);
fprintf('  near-unanimous across participants: %d   (|rank-biserial r| >= 0.9)\n', unanimous);
%%
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
%%
% *What the data say.* The kind of help is what decides speed, not the control
% mode. Averaged over both modes, force guidance alone (F) took about 290 s,
% blending (B) about 220 s and both together (FB) about 200 s: *blending saves
% roughly a minute and a half per trial*, and adding force on top of blending
% saves a further ~20 s (a small but significant step, B vs FB p = 0.014). This
% is the largest effect in the whole study: assistance explains three quarters
% of the variation in task time (|partial_eta2| = 0.74). Clutch and joystick,
% averaged over the three kinds of help, take the same time (232 s vs 241 s,
% p = 0.34).
%
% The interesting detail is that the two factors *interact* (p = 0.033): the
% *slowest cell of all is JF* (306 s), slower even than CF (273 s), while
% *JFB is the fastest* (196 s). Half of all JF trials ran over five minutes;
% one JFB trial did. The most plausible reading is mechanical: in joystick
% mode the handle's offset _is_ the velocity command, so a guidance force that
% displaces the handle is also steering the robot, and without blending to make
% the two channels agree the operator ends up negotiating with the guidance.
% Once blending is added they point the same way, and the joystick becomes the
% fastest configuration. Force guidance alone is a poor match for a velocity
% mapping; force guidance on top of blending is the best combination tested.
%
% Path length tells the same story with one addition: here the mode does
% matter too (joystick paths 12% shorter, 18 of 24 participants). From the
% worst cell to the best the grippers travel *40% less* (8.5 m in CF against
% 5.1 m in JFB), and the between-participant spread shrinks with it — the
% worst-performing operators gain the most.
%% 3b. Was it safer?
% *Minimum clearance (worst hand)* is the closest either arm came to an
% obstacle while the operator was driving. It is each arm's own clearance as
% the collision barrier itself sees it — distance to the other arm, the
% objects, the rack and the table — with two things deliberately left out: the
% autonomous grasp moments (the gripper is _supposed_ to touch the object
% then) and *the cylinder the arm is carrying*, which is fused to the gripper
% and is payload, not obstacle. The controller's raw "closest pair" number is
% not used, because it reads −3.5 cm the whole time a cylinder is held and says
% nothing about obstacles. The clearance is measured from the robot's safety
% envelope, a few centimetres outside the metal, and is a conservative (slightly
% pessimistic) estimate. Higher is better.
%
% *Safety filter active* is the fraction of the trial in which the robot's
% collision barrier actually had to push back against the command it was given.
% This is the honest measure of "how often did the operator need saving" —
% lower means they stayed out of trouble by themselves.

fig_paper_panels(S, items, METRICS(3:4), 'cols', 1, 'name', 'safety');
%%
% *What the data say.* *Minimum clearance is the same in every condition*:
% about 2 cm (1.9–2.0 cm), and no test finds a difference (p = 0.12). This is
% not the operator's doing — it is the *safety filter's*. The barrier is tuned
% to hold the arm about 1.5 cm plus a speed-dependent margin off the envelope,
% and in every trial, under every condition, the arm at some point pressed
% against that limit and was held there. The panel therefore says two useful
% things: *no condition ever got closer than the barrier allows* (there were no
% collisions or near-collisions in the study), and *how close the robot can
% get is decided by the controller, not by how the operator is helped*. The
% differences in safety behaviour are in the next panel and in section 6b.
%
% *Safety filter active is decided by the control mode* (|partial_eta2| = 0.75,
% the second-largest effect in the study): in clutch the barrier had to
% intervene about 32% of the time, in joystick about 25% — a fifth less, and
% 22 of 24 participants show it. The velocity mapping keeps the operator out of
% trouble; position mapping with a re-centring button does not.
%
% Assistance appears to run the *other* way — the filter is busiest under *FB*
% (31%) and least busy under *F* alone (27%) — but this is *the denominator
% moving, not the safety*. Converted to seconds (section 6b), the filter was
% active for about 79 s per trial under F, 62 s under B and 62 s under FB: *F
% alone had the most barrier time, not the least*. What blending removes is the
% other part of the trial — the free-space wandering, 211 s under F against
% 139 s under FB — so the same barrier time becomes a larger _share_ of a
% shorter trial. Any "fraction of the trial" quantity behaves this way when
% trials differ in length; the seconds are the fairer reading, and they say
% blending reduced both the total time and the time at the barrier.
%% 3c. Was the resulting motion better?
% *Smoothness (SPARC)* is a standard smoothness score for the hand's speed
% profile. It is always negative: *closer to zero means smoother*, more negative
% means the motion was jerky, hesitant or full of stop-and-go.
%
% *Commanded joint-rate RMS* is how vigorously the controller had to move the
% arm joints to follow the operator. Lower means calmer, less frantic motion.

fig_paper_panels(S, items, METRICS(5:6), 'cols', 1, 'name', 'motion quality');
%%
% *What the data say.* *Smoothness is the one quantity both factors improve, by
% the same amount and independently* (|partial_eta2| ≈ 0.52 each, no
% interaction). Joystick is smoother than clutch for 22 of 24 participants —
% every clutch re-centring is a stop in the speed profile, and SPARC counts
% them. Blending is smoother than force guidance alone, and F-only is the most
% hesitant condition in both modes. From CF (−9.7) to JFB (−7.2) the score
% improves by about a quarter; 11 of the 15 pairs differ significantly, the
% most of the six headline panels.
%
% *Joint-rate RMS is the panel to read most carefully.* There is no overall
% winner — neither mode nor assistance has a main effect — but a strong
% *crossover* (|partial_eta2| = 0.51): in clutch, F-only is the most demanding
% cell and B the calmest; in joystick it is the reverse, F-only is calmest and
% FB the most demanding. The likely reason is that under blending in joystick
% mode the robot adds its own motion on top of the operator's, so more joint
% motion is commanded per second — *the robot is doing more, not the operator*.
% The range is moderate (15% between the extreme cells). This is why the document
% does not present joint-rate demand as "effort" without qualification; the
% haptic work the operator actually absorbed is discussed in 6f.
%% 4. What the statistics say
% The figures above show *where* the differences are. This section says which of
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
% The answer to each question is a number called *partial eta squared*
% (|partial_eta2|), drawn as a bar below. It is *the share of a metric's
% variation that one factor explains*, from 0 (the factor changes nothing) to 1
% (the factor explains everything). Concretely: task time varies from trial to
% trial for many reasons — who the operator is, which scene, which condition,
% luck. Of the part that the conditions can explain at all, |partial_eta2| =
% 0.74 for Assistance means three quarters is down to which kind of help was
% given, and 0.04 for Mode means the mapping barely matters. Roughly, 0.01 is a
% small effect, 0.06 medium, 0.14 large; most values here are well past
% "large". It answers "how much does this matter", which the p-value cannot;
% *compare results with each other by this bar, not by their p-values.*
%
% A filled bar is a significant effect, a hollow one is not. The stars repeat
% the p-value in the usual shorthand.

fig_effect_map(S, items, METRICS, alpha);
%%
% *How to read the figure.* A clear division of labour appears:
%
% * *Assistance* owns the time-and-path quantities (0.74–0.77: enormous) and
% contributes to smoothness (0.53).
% * *Control mode* owns the safety filter (0.75) and contributes equally to
% smoothness (0.52); it has a moderate effect on path length (0.35) and none on
% task time.
% * *The interaction* — the case where the best help _depends on_ the mode —
% matters in two places only: joint-rate demand (0.51, the crossover described
% in 3c) and, more mildly, task time (0.16, the JF anomaly described in 3a).
% An interaction of 0.51 with no main effects, as for joint-rate demand, means
% "assistance changes this metric a great deal, but in opposite directions in
% the two modes, so on average it cancels out" — the two crossing patterns are
% the whole story, and an average over them is meaningless.
% * *Minimum clearance* has nothing in any column, for the reason given in 3b.
%
% Nothing in this figure needs the tables of the full report: the brackets in
% the panels already show _which pairs_ differ, and this figure shows _which
% factor_ is responsible. The pair-by-pair numbers are in |study_report| for
% anyone who wants them.
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
%%
% *What the data say.* The composite gives a clean answer: *JFB is the best
% condition and CF the worst*, 0.68 standard deviations apart, with the three
% joystick cells above the three clutch cells (JFB > JB > JF > CFB > CB > CF).
% Twelve of the fifteen pairs differ significantly. Joystick beats clutch for
% 22 of 24 participants; blending beats force-only decisively; FB beats B by a
% small but real margin (0.12, p = 0.01). The composite shows *no learning
% trend and no order bias*, so this ranking is not an artefact of who did what
% first.
%
% The three family panels show where that comes from and where it does not:
%
% * *Time & effectiveness* is entirely an assistance story (F-only cells at
% about −0.6, all others positive) — the two modes are level.
% * *Safety* is mostly a mode story (all three joystick cells above all three
% clutch cells, 20 of 24 participants). Assistance has a small effect here, and
% not in blending's favour: blending alone (B) scores best and blending with
% force (FB) worst, though only that one pair separates (p = 0.04). Section 6b
% explains why.
% * *Motion quality* is mostly an assistance story again, with the two F-only
% cells alone below zero.
%
% So "joystick is better" and "blending is better" are true for *different
% reasons*: the mapping decides how often the operator gets into trouble and how
% smoothly the arm moves; the assistance decides how quickly and directly the
% task gets done. The best cell is simply the one that has both.
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
% _within_ each mode in the full report (|study_report|); their result is
% summarised in words in 6f.
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
%%
% *What the data say.* Time under human control mirrors task time exactly (the
% JF anomaly included), which confirms the speed gain is in the operator's part
% of the task, not in the autonomous grasp. Path efficiency runs from 0.16 (CF)
% to 0.26 (JFB): even in the best condition the grippers travel about four
% times the straight-line distance — the task requires going around the rack,
% and both arms are counted — but the improvement is large (60%) and shared by
% both factors.
%
% Mean hand speed is the one panel where *clutch is higher* (0.029 vs
% 0.023 m/s, 21 of 24), and yet clutch is not faster at the task: speed without
% direction. Read together with path length this says clutch operators moved
% faster along longer, more indirect paths. Time in the autonomous grasp phases
% is shortest under FB (~38 s) and longest under F-only (46–52 s): the grasp
% routine finishes sooner when the arm arrives at a better pose, which is what
% blending arranges.
%% 6b. Safety — the rest
% All clearances here are the per-arm barrier clearance of 3b (carried cylinder
% and grasp moments excluded). *Mean clearance* is its time-average while the
% operator drove — how much room the arm typically kept, as opposed to the
% single closest moment. *Near-miss time fraction* is the share of that time
% spent closer than 5 cm to something, and *near-miss time* is the same in
% seconds — the pair is shown together because a fraction rises when a trial
% gets shorter even if the seconds do not. *Near-miss episodes* counts how many
% separate times either arm dipped below 5 cm: one long approach and ten brief
% scares look very different to an operator but can average the same.

fig_paper_panels(S, items, ["safety_mean_dist_m", "safety_nearmiss_frac", ...
                 "safety_nearmiss_s", "safety_nearmiss_episodes"], ...
                 'cols', 1, 'panel_h', 235, 'name', 'clearance and near misses');
%%
% *What the data say.* Mean clearance is 5.5–6.1 cm everywhere: the arm
% typically kept about three times the barrier's minimum off the envelope. It
% is slightly higher in joystick (16 of 24, a small effect) and, in clutch
% only, lowest under FB — the same pattern the fraction panel shows more
% strongly.
%
% The near-miss panels are the ones where the fraction and the seconds *tell
% different stories, and the seconds are right*. By fraction, CFB looks worst
% (57% of the drive within 5 cm, against 47% for CF) and blending looks like it
% brings the arm closer to things. By seconds, force guidance alone had the
% most near-miss time of all (about 117 s per trial, against 86 s under B and
% 84 s under FB), because those trials were long and the operator kept
% re-approaching. Blending did not add close time; it removed the far-away
% time — the free-space wandering fell from 124 s under F to 79 s under FB —
% so the close time became a larger share of a shorter drive. This is exactly
% the effect described for the safety filter in 3b, and it is why the
% assistance ranking of the safety score in section 5 (B > F > FB) should not
% be read as "blending is less safe": three of its five metrics are fractions
% or averages over the drive.
%
% The episode count is the metric least affected by trial length (it barely
% correlates with task time), and it is unambiguous: both factors reduce it,
% from 17 separate close approaches per trial under CF to 12.6 under JFB, with
% joystick beating clutch for 19 of 24 participants and every JB/JFB cell
% significantly below CF, CFB and JF. Read together: *blending replaces many
% brief close calls with fewer, longer, controlled close passes; the joystick
% mapping reduces both.*
%% 6b′. Safety — the filter itself
% *Safety filter active time* is the filter-active fraction of 3b converted to
% seconds. *Typical barrier push* is how hard the filter pushed when it was
% pushing — the median of its multiplier over the active samples. The plain
% mean of the multiplier is not shown: it is dominated by rare spikes (a hard
% barrier fight raises it a thousand-fold for a few seconds, and about one trial
% in eight contains one), so a single such episode would own the bar.

fig_paper_panels(S, items, ["cbf_active_s", "cbf_lambda_active_median"], ...
                 'cols', 1, 'panel_h', 250, 'name', 'safety filter');
%%
% *What the data say.* In seconds, the filter's picture is simple: joystick
% needs it less (53–75 s per trial against 70–83 s in clutch) and force
% guidance alone needs it most in both modes. *How hard* the filter pushed when
% it did is the same everywhere — a typical multiplier of 24–28 with no mode or
% assistance effect. So the conditions change *how often* the operator reached
% the barrier, not what happened there: once the arm is at the limit, the
% controller does the same job regardless of how the operator is being helped.
%% 6c. Motion quality — the rest
% *Mean tracking slack* is how far the robot was allowed to fall behind the
% reference it was given: large slack means the controller was relaxing the
% tracking constraint to stay feasible, usually because it was busy avoiding
% something. *Peak commanded joint rate* is the single fastest joint command in
% the trial — the spikes that a mean hides.

fig_paper_panels(S, items, ["slack_mean", "qdot_cmd_max"], ...
                 'cols', 1, 'panel_h', 250, 'name', 'motion quality (rest)');
%%
% *What the data say.* Tracking slack is *the one motion metric where clutch
% wins* (lower slack for 21 of 24 participants, |partial_eta2| = 0.62): a
% position mapping gives the controller a reference it can follow closely,
% whereas a velocity mapping integrates the handle offset and lets the
% reference run ahead of the arm. Assistance matters just as much (0.64), with
% F-only clearly worst in both modes — blending produces references the robot
% can actually track. Peak joint rate is dominated by CF, whose spikes (1.15
% rad/s) stand well above every other cell's (0.7–0.9); among those the
% differences are modest. The frantic moments belong to force guidance alone
% under position control.
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

fig_paper_panels(S, items, ["belief_mean_prob", "belief_max_prob", ...
                 "belief_time_to_conf_s"], 'cols', 1, 'panel_h', 235, 'name', 'intent understanding');
%%
% *What the data say.* The intent estimator worked in every one of the 288
% trials: it always became confident, and its *peak* confidence was above 0.99
% everywhere. That makes peak confidence *the textbook example of a significant
% but unimportant result*: joystick beats clutch with p < 0.001 and all 24
% participants in agreement — by 0.003 in probability (0.997 against 0.994;
% note the clipped axis). Real, and of no consequence: the robot was certain in
% both cases.
%
% *Mean* confidence is the informative one, because it also counts how long the
% estimator stayed unsure after each reset. It runs at about 0.77 in clutch and
% 0.81 in joystick — a large mode effect (|partial_eta2| = 0.60, 20 of 24
% participants) with assistance making no difference. The reading: the steady
% velocity of a joystick approach is easier to interpret than a position path
% broken up by re-centrings, so the estimator spends less of the trial
% hedging. Time to the first confident estimate (4.5–8.8 s) shows no reliable
% difference between conditions (p = 0.19), which is as it should be — the help
% offered downstream should not change how quickly the goal is recognised
% upstream.
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
%%
% *What the data say — and a warning about what they are.* The first three
% panels *describe the assistance rather than judge the operator*. Authority and
% autonomy-led time are what the blending law decided to do, given the intent
% estimate and the mode; they are marked "neither better nor worse" for that
% reason. Their p < 0.001 brackets say the _system_ behaved differently in
% joystick mode — the blending law was granted about a third of the motion
% (0.31) in JFB against a fifth (0.20) in CB — not that the operator performed
% better. Do not count these among the study's wins.
%
% The panel that does carry judgement is *agreement*: how often operator and
% robot pulled the same way. It is highest under JFB (0.35) and lowest under CB
% (0.12), every one of the six pairs differs, and the two factors *reinforce
% each other* (interaction p < 0.001): adding force guidance to blending
% improves agreement roughly twice as much in joystick mode as in clutch. This
% is the mechanism behind the JFB result — force feedback tells the operator
% where blending is about to take them, and in a velocity mapping the two act
% through the same channel. The absolute values are modest in every condition
% because the average includes every moment of the trial, including those where
% the operator is repositioning and not heading to the goal at all; read the
% panel relatively.
%
% *User actively driving* is flat (67–69%, no differences): operators were
% commanding about two thirds of the time whatever the condition. The
% assistance changed what happened when they commanded, not how much they did.
%% 6f. The remaining family scores
% The same z-scored, higher-is-better summaries as section 5, for the three
% families not shown there. *Human effort* here rests on the joint-rate demand
% only, for the cross-mode reason given at the top of this section.

fig_paper_panels(S, items, ["fam_human_effort", "fam_intent_understanding", ...
                 "fam_assistance_quality"], 'cols', 1, 'panel_h', 235, 'name', 'remaining family scores');
%%
% *What the data say.* *Human effort*, as this chart is forced to define it,
% has no winner: the crossover of 3c reappears (JF best, CF worst, the rest
% level) and neither factor has a main effect. The operator's haptic effort,
% which can only be compared within a mode, is unambiguous in the full report:
% in clutch, blending roughly halves the mean handle force (0.48 N against
% 0.82 N under F-only), halves the force impulse (104 against 232 N·s) and cuts
% clutch presses from 22 to 16 per trial; in joystick the impulse falls from
% 480 N·s under F-only to 279 N·s under FB. *Blending halves the physical work
% the operator absorbs from the handle, in both modes.* That is the effort
% result of this study; the joint-rate panel is not.
%
% *Intent understanding* favours joystick clearly (21 of 24, a large effect
% now that mean confidence is in the score) and is indifferent to the kind of
% help — consistent with 6d. Its practical weight is modest: the estimator
% succeeded everywhere, it was simply surer for longer with a joystick.
%
% *Assistance quality* (agreement) is the most emphatic family: JFB stands 1.9
% standard deviations above CB, all six pairs differ, and the mode explains 83%
% of the variation. Keep the caveat of 6e in mind: this measures how well the
% help and the operator lined up, which is a precondition for the help being
% useful rather than a performance outcome in itself.
%% 7. What this page does not tell you
% * *The mode comparison on time-based quantities is partly a practice effect.*
% Participants improved over their twelve trials — task time fell by about 11 s
% per trial slot, a large learning effect — and because each participant did
% all clutch trials then all joystick trials (or the reverse), whichever mode
% came second benefited from that practice. The analysis flags this: on task
% time, time under human control and the time-effectiveness score — and, more
% weakly, on the safety-filter fraction, joint-rate demand and agreement — the
% joystick-minus-clutch difference depends on which mode came first. The
% design balanced the order (12 and 12), so the _average_ is fair, but the
% mode effect on speed should be read as "none detected", which is also what
% the test says. *The composite and the safety, motion-quality and
% intent-understanding family scores show no order bias*, so the claim
% "joystick is better" rests on those, not on speed.
% * *The shield scene was easier than the rack scene* on nearly every measure
% (composite: 24 of 24 participants). Every participant did both scenes under
% every condition, so this does not bias any comparison above; it does mean the
% absolute numbers are an average of an easy and a hard scene.
% * *Several metrics grow with trial length, and the conditions differ in
% length.* Force guidance alone took ~45% longer than FB. Anything that
% accumulates over a trial inherits that: hand path length (correlation with
% task time 0.76), haptic force impulse (0.60), time in autonomous grasp
% phases (0.59), clutch presses (0.35), and the seconds-based safety
% quantities. The opposite bias exists too: any *fraction of the trial* rises
% when the free-space part of the trial shrinks (sections 3b and 6b show this
% for the safety filter and the near misses). SPARC smoothness also correlates
% with task time (−0.61), partly because a longer drive contains more
% sub-movements, so part of its assistance effect is the time effect seen
% again. Metrics that are neither counts nor fractions — mean clearance,
% typical barrier push, tracking slack, joint rates, time to confident intent
% — carry no such bias, and the episode count barely does (0.12). The
% conclusions above lean on the biased quantities only where a time-free one
% agrees with them.
% * *Minimum clearance is set by the controller, not the operator.* About 2 cm
% in every condition is the barrier holding the arm at its limit; the metric
% confirms no condition breached it, and cannot separate conditions because
% the limit is the same in all of them. It is the safety envelope that is
% measured, a few centimetres outside the metal.
% * *The barrier multiplier is heavy-tailed everywhere*, not in one trial. About
% one trial in eight contains a spike of a thousand or more (the largest, P18
% rack JF, reaches eight million for fifteen seconds during a hard barrier
% fight). Those are real events, not corrupt data — the slack and joint rates
% of the same seconds confirm the arm was genuinely stuck against the barrier
% — so no trial is dropped; the panel uses the median push when active, which
% those seconds cannot dominate, and the plain mean is kept only as a
% diagnostic.
% * *Only objective measures are shown.* What the operators _felt_ — workload,
% trust, preference — is collected through the post-trial questionnaire and is
% not part of this document. In the paper this study follows, that subjective
% panel is where the conditions sometimes disagree with the objective ones, so
% it should be read alongside this page before drawing conclusions.
% * *The experimenter's success/failure notes are not analysed.* They were
% recorded inconsistently (five trials are marked unsuccessful, three of them
% without a reason) and every trial passed the completeness check, so no
% success-rate comparison is made.
% * *Two quantities are deliberately absent from the charts*: the haptic force
% and the clutch button, for the apparatus reason given in section 6; their
% within-mode results are summarised in 6f.
% * *"Not significant" is not the same as "no difference"* — but with 24
% participants in a within-person design, effects of moderate size are detected
% reliably, and where this study reports no difference (minimum clearance,
% typical barrier push, time to confident intent, user activity) the estimated
% gaps are close to zero. Those are most likely genuine nulls, not missed
% effects.

fprintf('\nBuilt from %s\n', RESULTS_DIR);
