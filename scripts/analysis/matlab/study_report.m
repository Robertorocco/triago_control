%% TRIAGo shared-autonomy user study: group-level statistical report
% This Live Script is the readable report of the analysis produced by
% |run_study_analysis.m|. It explains what every performance metric means,
% how every statistical test works and how to read every figure, and then
% answers the study questions on the participants analysed so far.
%
% *How to use it.* Run |run_study_analysis| first (it writes a time-stamped
% results folder). Then run this script: by default it loads the most recent
% results folder and redraws every figure. Nothing needs to be edited when new
% participants are added: re-run the analysis, then re-run (or re-build) this
% report.

%% Settings
RESULTS_DIR = '';               % '' = most recent run (analysis_results/latest.txt)
EXPORT_DIR  = '';               % '' = auto-locate manifest.mat (see study_export_dir)
SHOW_ALL_METRIC_FIGURES = true; % false = only one headline metric per family
%% Load the results
if isempty(RESULTS_DIR)
    root = fullfile(study_export_dir(EXPORT_DIR), 'analysis_results');
    RESULTS_DIR = strtrim(fileread(fullfile(root, 'latest.txt')));
end
load(fullfile(RESULTS_DIR, 'results.mat'), 'trial', 'spec', 'families', 'fam', 'meta', 'S', 'R', 'items', 'settings');
fprintf('results folder : %s\nrun time       : %s\nparticipants   : %d complete (%s)\n', ...
    RESULTS_DIR, string(settings.run_time), meta.n_complete, strjoin(meta.complete_list, ', '));
disp(meta.excluded)
%% 1. The study in one paragraph
% Each participant teleoperated the TRIAGo robot through a pick-and-place task
% with the Haption haptic device under every combination of two factors:
%
% * *Control mode* (2 levels): *Clutch* (C, position control: the handle
% displacement is integrated into a pose reference; a clutch button re-indexes
% the workspace) and *Joystick* (J, velocity control: the handle displacement
% from a spring-centred home pose commands a velocity).
% * *Assistance* (3 levels): *F* = assistive haptic feedback (guidance forces on
% the handle), *B* = reference blending (the robot mixes the operator's command
% with its own policy toward the inferred goal), *FB* = both.
%
% The six cells are named by their letters: CF, CB, CFB, JF, JB, JFB. Every
% cell was run in two scenes (*worlds*): _rack_ (a shelf forces side grasps
% and a two-arm hand-over) and _shield_. Each participant therefore did 12
% trials. The order of the six cells was counter-balanced with a schedule
% (|participant_schedule.csv|): the three cells of one control mode first,
% then the three of the other; within a cell the shield world was always run
% before the rack world.
%
% *Design type.* Every factor is *within-subject* (each person did every
% condition), so every comparison below is *paired*: a condition is compared
% with another condition _inside the same person_, which removes the large
% differences between people from the comparison.
%
% *Who is analysed.* Only participants with all 12 trials present. The
% |load_study_table| rule is automatic: a participant must be in the schedule
% and have the 12 (world, cell) folders. The excluded ones and the reason are
% listed above.

%% 2. How to read a statistical result
% *p-value.* The probability of seeing a difference at least as large as the
% observed one if, in reality, the conditions were identical. Small p (below
% the significance level $\alpha = 0.05$) means the data are hard to explain
% by chance alone: the difference is called *significant*. A p-value above
% 0.05 does *not* prove that the conditions are equal; with 12-24 participants
% only medium-to-large effects can be detected, so "not significant" means
% "not demonstrated with this sample", nothing more.
%
% *Effect size.* How large the difference is, independently of the sample
% size. Reported here:
%
% * *Cohen's* $d_z = \bar{d} / s_d$ for paired comparisons: the mean of the
% within-person differences divided by their standard deviation. Bands: 0.2
% small, 0.5 medium, 0.8 large.
% * *Rank-biserial* $r$ for the Wilcoxon test: $r = (W^+ - W^-) / (W^+ +
% W^-)$ where $W^\pm$ are the sums of the ranks of the positive / negative
% differences; $r = 1$ when every participant goes the same way. Bands: 0.1
% small, 0.3 medium, 0.5 large.
% * *Partial eta squared* $\eta_p^2 = SS_{effect} / (SS_{effect} + SS_{error})$
% for ANOVA: the share of the variance (after removing the participant
% differences) explained by the factor. Bands: 0.01 small, 0.06 medium, 0.14
% large.
% * *Kendall's W* $= \chi^2_F / (n(k-1))$: agreement of the $n$ participants on
% the ranking of the $k$ conditions, from 0 (no agreement) to 1 (everyone ranks
% them identically). Bands: 0.1 weak, 0.3 moderate, 0.5 strong.
%
% *Confidence interval (CI).* The range in which the true mean difference
% plausibly lies (95%). A CI that does not contain 0 corresponds to a
% significant difference. Where the sample is small, a *bootstrap* CI is used:
% the participants are re-sampled with replacement 5000 times and the 2.5th
% and 97.5th percentiles of the re-computed means are taken.
%
% *Two tests for every comparison.* A parametric test (t-test, ANOVA) assumes
% roughly normal differences; a non-parametric test (Wilcoxon, Friedman) only
% uses the ordering of the values and is safer with few participants and
% outliers. Both are always computed; a normality check (Lilliefors test on
% the differences) and, for ANOVA, a sphericity check (Mauchly) decide which
% one is *recommended* and used for the headline p-value. When the two
% disagree, the conclusion is fragile and is said so.
%
% *Multiple comparisons.* Testing many metrics inflates the chance that some
% p-value falls below 0.05 by luck. Two corrections are applied with the
% Holm-Bonferroni procedure (sort the p-values, multiply the smallest by $m$,
% the next by $m-1$, ..., keep them monotone): (1) inside one test, over the
% pairwise post-hoc comparisons (3 pairs for F/B/FB, 15 pairs for the six
% cells); (2) across the metrics of one family within one question. Both the
% raw and the corrected p-value are kept in the tables. The *family scores*
% (Section 4) are the headline answers and are reported uncorrected because
% there is only one of them per family.
%
% *Best.* Every metric has a direction (lower task time is better, higher
% clearance is better). "Best" always means better in that direction; metrics
% without a direction (mean speed, authority share) are shown for context only.

%% 3. The metrics
% Every trial is recorded as a ROS bag; |study_metrics.py| turns it into the
% numbers below (one value per trial, per arm when the quantity is arm
% specific). The table is the machine-readable specification used by every
% script; the paragraphs after it give the formulas.
disp(spec(spec.family ~= "", {'name', 'label', 'unit', 'dir', 'family', 'cell_scope', 'mode_comparable'}))
%% 3.1 Combining the two arms
% The robot has two arms and only one is teleoperated at a time (the other is
% frozen by the controller). Quantities measured on one arm are combined into
% one value per trial as follows, where $w_R$ and $w_L$ are the fractions of
% the trial during which the right / left arm was the active one:
%
% * *shared*: trial-level quantities (time, safety, force, clutch, belief,
% blending) are taken once; the export writes them identically in both arm
% columns.
% * *sum*: extensive quantities add over the arms: path length $L = L_R +
% L_L$, straight displacement $D = D_R + D_L$.
% * *weighted mean*: intensive quantities are averaged with the activity
% weights, $x = (w_R x_R + w_L x_L) / (w_R + w_L)$ (speed, smoothness,
% joint-rate RMS, slack, barrier multiplier, safety-filter activity).
% * *max*: worst-case quantities take the larger of the two arms (peak joint
% rate).
%
% Path efficiency is the ratio of the sums, $E = D / L$, not a mean of two
% ratios, so an idle arm cannot inflate it.

%% 3.2 Time and effectiveness
% * *Task time* $T = t_{last} - t_{first}$: span of the recorded messages of
% the trial, in seconds. Lower is better.
% * *Time in autonomous grasp phases* $T_g = \sum_i \Delta t_i \, g_i$ where
% $g_i \in \{0,1\}$ is the |grasp_active| flag (robot-driven approach, close,
% lift, release). Lower is better: a shorter autonomous phase means the
% operator brought the gripper to a good pre-grasp pose.
% * *Time under human control* $T_h = T - T_g$. Lower is better.
% * *Hand path length* $L = \sum_i \| p_{i+1} - p_i \|$, the arc length of the
% end-effector position samples $p_i$ (summed over the arms). Lower is
% better: a long path means detours, corrections, hesitation.
% * *Path efficiency* $E = \|p_N - p_1\| / L$ (ratio of the arm sums), between
% 0 and 1. Higher is better. Note that the task itself is not a straight line
% (pick, hand over, place), so $E$ is well below 1 even for a perfect run; it
% is meaningful as a comparison between conditions, not as an absolute.
% * *Mean hand speed* $\bar v = \mathrm{mean}_i \| v_i \|$ of the active arm.
% No direction: fast is not better if it is jerky.

%% 3.3 Human effort
% * *Commanded joint-rate RMS* $\sqrt{\mathrm{mean}_{i,j} \dot q_{ij}^2}$ over
% the samples $i$ and the 7 joints $j$ of the active arm, from the velocity
% command produced by the safety QP. Lower is better: less joint motion for
% the same task.
% * *Haptic force* $f_i = \| F_i \|$, the magnitude of the wrench rendered on
% the handle: *mean* $\mathrm{mean}_i f_i$, *peak* $\max_i f_i$ and *impulse*
% $I = \int f \, dt$ (trapezoidal rule). Lower is better (less load on the
% operator's hand). *Caution:* the force has a different physical origin in
% the two modes (a tether that pulls the handle toward the robot pose in
% Clutch, a centring spring in Joystick), so the force metrics are compared
% *only within one control mode*, never Clutch vs Joystick.
% * *Clutch presses*: number of rising edges of the clutch button; *clutch
% duty*: fraction of samples with the button held. Lower is better (fewer
% re-indexings of the workspace). Defined in Clutch mode only.

%% 3.4 Safety
% The controller publishes at every tick the signed distance $d(t)$ between
% the closest pair of collision bodies (robot links, objects, table, shelf).
% Samples inside the autonomous-grasp window are excluded, because the
% intentional gripper-object overlap during a grasp would otherwise read as a
% collision.
%
% * *Minimum clearance* $d_{min} = \min_t d(t)$ over the teleoperated samples.
% Higher is better.
% * *Near-miss time fraction*: share of teleoperated samples with $d(t) <
% 0.05$ m. Lower is better.
% * *Near-miss episodes*: number of separate dips of $d(t)$ below 0.05 m
% (rising edges of the indicator). Lower is better.
% * *Safety filter active*: fraction of samples in which the Lagrange
% multiplier $\lambda$ of the collision barrier constraint exceeds 1, i.e. the
% safety filter is actively altering the commanded motion. Lower is better:
% the operator kept the robot away from the barrier by themselves.
% * *Mean barrier multiplier* $\bar\lambda$: the average of $\lambda$. Lower is
% better: it measures how hard the filter had to push against the command.

%% 3.5 Motion quality
% * *Smoothness (SPARC)*, the spectral arc length of the hand speed profile
% $v(t)$ (Balasubramanian et al. 2015): with $\hat V(\omega)$ the magnitude
% spectrum of $v(t)$ normalised to its peak and restricted to $[0, \omega_c]$,
% $\omega_c = 10$ Hz,
%
% $$ \mathrm{SPARC} = - \int_0^{\omega_c} \sqrt{ \left( \frac{1}{\omega_c} \right)^2 + \left( \frac{d \hat V(\omega)}{d \omega} \right)^2 } \, d\omega $$
%
% i.e. minus the length of the normalised spectrum curve (the band is adapted
% to where the spectrum exceeds 5% of its peak). A smooth movement has a
% compact spectrum and a short curve, so SPARC is close to 0; a jerky movement
% spreads energy over frequencies and SPARC becomes more negative. *Higher
% (closer to 0) is better.*
% * *Mean tracking slack*: the safety QP tracks the operator's reference with
% a control-Lyapunov constraint that is relaxed by a slack variable; the mean
% slack tells how much the reference could not be followed (obstacle, joint
% limit, speed limit). Lower is better.
% * *Peak commanded joint rate* $\max_{i,j} |\dot q_{ij}|$. Lower is better
% (no violent joint motions).

%% 3.6 Intent understanding
% The shared-autonomy node maintains a probability $P_k(t)$ for every
% candidate goal $k$ (which object, which grasp side, which placement).
%
% * *Peak intent confidence* $\max_t \max_k P_k(t)$. Higher is better: the
% robot became sure of what the operator wanted.
% * *Time to confident intent*: the first $t$ with $\max_k P_k(t) \ge 0.80$,
% NaN if never reached. Lower is better. It is measured from the start of the
% recording, so it also contains the time the operator needed to start moving
% toward an object.
% * *Intent ever confident*: 1 if the 0.80 level was reached at least once.

%% 3.7 Assistance quality (blending cells only)
% In the B and FB cells the reference sent to the robot is
% $v_{blend} = (1-\alpha) v_{user} + \alpha \, v_{policy}$, where $v_{policy}$
% is the robot's own twist toward the most likely goal and $\alpha \in [0,1]$
% is the authority given to the robot (large when the operator moves in the
% same direction as the policy).
%
% * *User-autonomy agreement*: mean over the samples in which the operator is
% moving of $\cos\theta = \langle v_{user}, v_{policy} \rangle / (\|v_{user}\|
% \|v_{policy}\|)$. +1 means the operator and the robot pull in the same
% direction, 0 orthogonal, -1 opposite. Higher is better.
% * *Mean autonomy authority* $\bar\alpha$, *autonomy-led time fraction*
% (share of samples with $\alpha > 0.5$) and *user actively driving* (share of
% samples with a non-zero user twist): context, no direction.

%% 3.8 Success and incidents
% * *Task success*: the experimenter's manual yes/no call at the end of the
% trial (1 = yes).
% * *Incident noted*: 1 when the experimenter wrote a note (fallen object,
% failed grasp, unreachable object, ...). It captures problems that did not
% prevent an eventual success. Both are rates near the ceiling / floor, so
% they are described by counts rather than tested.

%% 4. Family scores and the composite
% To answer "which condition is best" across many metrics without counting
% each of them separately, the metrics are grouped in families
% (|study_families|) and every metric is standardised:
%
% $$ z = \mathrm{dir} \cdot \frac{x - \mu}{\sigma} $$
%
% where $\mu, \sigma$ are the mean and standard deviation of the metric over
% *all* trials of all complete participants and $\mathrm{dir} = \pm 1$ is the
% direction of the metric, so that *higher z always means better*. The
% *family score* of a trial is the mean of the z-scores of the family's
% metrics; the *composite* is the mean of the family scores. A score of 0 is
% the average trial of the study, +1 is one standard deviation better.
%
% Only metrics that are defined in every cell and comparable across modes
% enter a family score (force and clutch metrics do not; the assistance
% quality family exists only in blending cells). Metrics that never vary are
% dropped automatically. Family scores are also tested with exactly the same
% procedures as the metrics and are the headline answer per family; the
% composite is a descriptive summary, useful as one picture, not as a proof.
disp(families(:, {'key', 'label'}))
disp(fam(:, {'key', 'n_metrics', 'members', 'dropped'}))

%% 5. The tests, formally
% All tests are run on one value per participant per condition (the mean over
% that participant's trials in that condition), so $n$ is the number of
% participants.
%
% *Two conditions* (Clutch vs Joystick, B vs FB, rack vs shield): with $d_i =
% x_i - y_i$, the paired t-test uses $t = \bar d / (s_d / \sqrt n)$ with $n-1$
% degrees of freedom; the Wilcoxon signed-rank test ranks $|d_i|$ and compares
% the rank sum of the positive differences with its distribution under the
% hypothesis of symmetric differences (exact distribution for $n \le 15$); the
% sign test only counts how many participants went each way.
%
% *Three conditions* (F, B, FB): the one-way repeated-measures ANOVA models
% $x_{ij} = \mu + \pi_i + \tau_j + \epsilon_{ij}$ (participant $i$, condition
% $j$) and tests $\tau_j = 0$ with $F = MS_{condition} / MS_{error}$, the error
% being the participant-by-condition interaction. The sphericity assumption
% (equal variances of all pairwise differences) is checked with Mauchly's test
% and the degrees of freedom are corrected with the Greenhouse-Geisser
% $\epsilon$. The Friedman test ranks the 3 values of every participant and
% tests whether the rank sums differ: $\chi^2_F = \frac{12}{nk(k+1)} \sum_j
% R_j^2 - 3n(k+1)$ (with tie correction). Post-hoc: the three pairwise
% Wilcoxon tests, Holm-corrected.
%
% *Six cells* (mode x assistance): the two-way repeated-measures ANOVA
% decomposes the cell means into a *mode* main effect, an *assistance* main
% effect and their *interaction* (does the effect of assistance depend on the
% mode?). Every effect has its own error term (participant x effect) and its
% own Greenhouse-Geisser correction. When the interaction is significant the
% simple effects (assistance within each mode, mode within each assistance)
% are reported. The Friedman test over the six cells answers "do the cells
% differ at all" and the 15 pairwise Wilcoxon tests with Holm correction
% identify which cells differ.
%
% *Consistency* (which condition gives the most similar results across
% participants): for every condition the standard deviation across
% participants of the participant means is computed (with a bootstrap CI). Two
% conditions are compared with the Pitman-Morgan test for equal variances of
% paired samples: with $u_i = x_i + y_i$ and $v_i = x_i - y_i$, the
% correlation $r_{uv}$ is zero if and only if $\mathrm{Var}(x) =
% \mathrm{Var}(y)$, and $t = r_{uv} \sqrt{n-2} / \sqrt{1 - r_{uv}^2}$ with
% $n-2$ degrees of freedom. Kendall's W adds the complementary notion: do the
% participants *agree on the ranking* of the conditions?
%
% *Learning and order*: for every participant the metric is averaged over the
% two worlds of every slot (6 values) and a least-squares line is fitted; the
% slope (change per slot) is tested against zero over the participants (t-test
% and Wilcoxon). The block effect compares slots 1-3 with slots 4-6. Because
% control mode was blocked, the Joystick-minus-Clutch difference of the
% participants who started with Clutch is compared with that of the
% participants who started with Joystick (Mann-Whitney rank-sum test): if
% these differ, part of the "mode effect" is practice.

%% 6. Overview: which condition is best, per family
% One figure per family: the six cells (colour and number = mean family
% score, red = better), then Clutch vs Joystick and F / B / FB with one grey
% line per participant. The titles carry the test results. The wide
% single-page version is figures/overview_families.png in the results folder.
for k = find(items.is_family & items.name ~= "composite")'
    fig_family_compact(trial, S, table2struct(items(k, :)), families);
end
%% 6.1 Headline table (family scores)
show_results(R(R.is_family_score & ismember(R.question, ["Q1_mode" "Q2_assist" "Q3_cell"]), :), "headline")

%% 7. Q1: Which teleoperation strategy is best, in general?
% Clutch versus Joystick, paired within participant, averaged over the three
% assistance levels and the two worlds. Family scores first, then every
% metric (Holm-corrected within its family).
show_results(R(R.question == "Q1_mode", :), "Q1")

%% 8. Q2: Which assistive strategy is best, in general?
% F versus B versus FB, averaged over the two modes and the two worlds. Force
% and clutch metrics are compared within one mode at a time; the assistance
% quality metrics only exist for B and FB.
show_results(R(R.question == "Q2_assist", :), "Q2")

%% 9. Q3: Which mode + assistance combination is best, in general?
% The six cells. The composite ranking is the one-picture summary; the family
% table and the per-metric table give the rigorous answers, including the
% interaction (whether the best assistance depends on the mode).
fig_cell_ranking(trial, S, 'compact', true);
show_results(R(R.question == "Q3_cell", :), "Q3")
%% 9.1 Mode x assistance interaction, per family
% An interaction means "the effect of assistance is different in Clutch and in
% Joystick". If it is significant, read the cell grid rather than the two
% main effects.
for nm = items.name(items.is_family)'
    if isfield(S.Q3, nm) && ~isempty(S.Q3.(nm).effects)
        E = S.Q3.(nm).effects;
        fprintf('%s\n', nm);
        for i = 1:height(E)
            fprintf('   %-12s F(%g, %g) = %6.2f   p (GG) = %.3f   partial eta2 = %.2f %s\n', ...
                E.effect(i), E.df1(i), E.df2(i), E.F(i), E.p_gg(i), E.eta2p(i), studyplot.stars(E.p_gg(i)));
        end
    end
end

%% 10. Q4 and Q5: Which strategy gives the most similar results among participants?
% Shorter bar = the participants obtained more similar values under that
% condition. A significant Pitman-Morgan test means the spread genuinely
% differs; Kendall's W says whether participants agree on the ranking.
fig_consistency_overview(S, items, families, 'compact', true);
show_results(R(R.question == "Q4_consistency_mode", :), "Q4 (best = most consistent)")
show_results(R(R.question == "Q5_consistency_assist", :), "Q5 (best = most consistent)")

%% 11. Q6: Learning and order effects
% Left: family score along the six slots. Right: the Joystick-minus-Clutch
% difference of each participant, split by which mode they met first. If the
% two groups differ, the mode comparison of Q1 is partly a practice effect and
% must be read with care.
fig_learning_overview(S, items, families, 'compact', true);
show_results(R(R.question == "Q6_learning", :), "Q6 (statistic = mean slope per slot)")

%% 12. Supplementary: the two worlds
% Rack versus shield. Every participant did both worlds in every cell, so the
% world effect cancels out of the paired comparisons of Q1-Q3; it is shown
% because it is large and tells how demanding each scene was.
fig_world_overview(trial, S, items, families, 'compact', true);
show_results(R(R.question == "QW_world" & R.is_family_score, :), "world")

%% 13. Success and incidents
fig_success_incidents(trial, 'compact', true);
disp(trial(trial.incident == 1, {'participant', 'world', 'cell', 'success', 'notes'}))

%% 14. Metric dashboards
% One figure per metric, four stacked panels: Q1 mode, Q2 assistance, Q3
% cells, Q6 trend. Grey lines are participants, coloured markers are means
% with 95% CI, brackets mark Holm-significant pairs. Larger versions of the
% same figures are in figures/metric_<name>.png in the results folder.
headlineMetrics = ["duration_s" "qdot_cmd_rms" "safety_min_dist_m" "ee_sparc" "belief_max_prob" "agreement_mean_cos" "composite"];
metricItems = items(~items.is_family | items.name == "composite", :);
if ~SHOW_ALL_METRIC_FIGURES
    metricItems = metricItems(ismember(metricItems.name, headlineMetrics), :);
end
for k = 1:height(metricItems)
    fig_metric_dashboard(trial, table2struct(metricItems(k, :)), S, 'compact', true);
end

%% 15. Assumptions, limitations, and how to judge a result
% * *Sample size.* With $n$ participants the paired tests detect an effect of
% $d_z \approx 0.9$ at $n = 12$ and $d_z \approx 0.6$ at $n = 24$ with 80%
% power. Medium effects that are not significant now may become significant
% when the study is complete; the effect sizes and CIs are the quantities to
% watch, not the p-values alone.
% * *Normality and sphericity.* The parametric results are only trusted when
% the Lilliefors and Mauchly checks pass; otherwise the rank-based test is the
% headline. When both tests agree the conclusion is robust.
% * *Multiple testing.* Per-metric significance is Holm-corrected within its
% family and question; a result that is significant before correction but not
% after is suggestive, not established.
% * *Ceiling effects.* Success is almost always "yes" and intent confidence is
% almost always reached, so those metrics carry little information; the
% incident notes and the time-to-confidence are more informative.
% * *Order.* Control mode was blocked (3 cells, then 3 cells). The
% counter-balancing across participants removes the average practice effect
% from the mode comparison, but only when both orders are equally represented;
% Section 11 checks this directly.
% * *World order.* Within a slot the shield world always preceded the rack
% world, so the two worlds cannot be compared for practice; slots are averaged
% over the worlds before any learning analysis.
% * *Mode-specific metrics.* Haptic force and clutch use are never compared
% across control modes.
% * *Belief metrics* depend on the number of candidate goals, which differs
% between worlds; they are compared across conditions within the same
% world-balanced design, never across worlds in absolute terms.

%% 16. Appendix: full results table and provenance
% The complete master table (every test, every metric) is results_all.csv in
% the results folder; a compact view:
A = compact_table(R); A.question = extractBefore(R.question + "    ", 3); A = movevars(A, 'question', 'Before', 'metric');
disp(A)
fprintf('manifest      : %s\nschedule      : %s\nresults       : %s\nalpha = %.2f, bootstrap resamples = %d\n', ...
    meta.export_dir, meta.schedule_path, RESULTS_DIR, settings.alpha, settings.n_boot);
%%
% *To update with new participants:* export them (|export_to_matlab.py|),
% copy the new |manifest.mat| into the bundle folder, run
% |run_study_analysis|, then |build_report|.

%% Local functions
function show_results(Rq, what)
% Compact table of the numbers, then the interpretation of every row as text.
% Family scores (the headline) come first, then the metrics sorted by
% family and corrected p-value.
Rq = sortrows(Rq, {'is_family_score', 'family', 'p_holm'}, {'descend', 'ascend', 'ascend'});
fprintf('--- %s: %d results ---\n', what, height(Rq));
disp(compact_table(Rq))
fprintf('Interpretation (family scores in CAPITALS are the headline answers):\n');
for i = 1:height(Rq)
    lab = Rq.label(i); if Rq.is_family_score(i), lab = upper(lab); end
    star = ""; if Rq.significant_holm(i), star = " [significant after Holm]"; end
    fprintf('\n* %s (%s)%s\n%s\n', lab, Rq.levels(i), star, wrap(Rq.interpretation(i), 92, "    "));
end
end

function T = compact_table(Rq)
T = table();
T.metric = extractBefore(pad("[" + extractBefore(Rq.family + "    ", 5) + "] " + Rq.label, 32), 33);
T.n = Rq.n;
T.p_raw = arrayfun(@ptxt, Rq.p_raw);
T.p_holm = arrayfun(@ptxt, Rq.p_holm);
T.effect = arrayfun(@(v) sprintf("%.2f", v), Rq.effect);
T.best = Rq.best;
end

function s = ptxt(p)
if isnan(p), s = "n/a"; elseif p < 0.001, s = "<0.001"; else, s = sprintf("%.3f", p); end
end

function out = wrap(str, width, indent)
% Word-wrap a sentence to WIDTH characters per line, each line prefixed by INDENT.
words = split(string(str), " ");
lines = strings(0, 1); cur = "";
for w = words'
    if strlength(cur) == 0
        cur = w;
    elseif strlength(cur) + 1 + strlength(w) > width
        lines(end + 1, 1) = cur; cur = w; %#ok<AGROW>
    else
        cur = cur + " " + w;
    end
end
lines(end + 1, 1) = cur;
out = strjoin(indent + lines, newline);
end
