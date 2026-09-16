%% TRIAGo shared-autonomy user study: group-level statistical report
% This Live Script is the readable report of the analysis produced by
% |run_study_analysis.m|. It explains what every performance metric means,
% how every statistical test works and how to read every figure, and then
% answers the study questions on the full set of complete participants.
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
% Every test in this report answers exactly *two questions*, always in this
% order:
%
% # *Is this difference real, or could it just be luck?* -- the *p-value*.
% # *If it is real, how big is it -- does it actually matter?* -- the
% *effect size*.
%
% Neither question alone is enough. A p-value alone can call a tiny,
% meaningless gap "real" if there is enough data; an effect size alone can
% call a huge number "important" when it is really just noise from too few
% participants. This report always gives both, plus a plausible range (the
% confidence interval) for how big the true gap is.
%
% *If you read nothing else, read this table:*
disp(cell2table({ ...
 "p < 0.05  (marked * ** ***)"                 "The gap is unlikely to be pure luck -- treat it as real."
 "p >= 0.05  (marked n.s.)"                    "Not enough evidence with this many participants. This does NOT mean the conditions are equal -- it means we could not tell them apart."
 "effect size: small"                          "Real, but modest in everyday terms."
 "effect size: medium / large"                 "A gap big enough that a person would actually notice or care about it."
 "95% CI does not include 0"                    "Same conclusion as p < 0.05, plus a plausible range for how big the true gap is."
 "the two calculators disagree (recommended flips)" "The result is fragile / borderline -- read it with extra caution."
}, 'VariableNames', {'if_you_see', 'it_means'}))

%% 2.1 "Is it real?" -- the p-value
% Picture a courtroom. The starting assumption ("innocent until proven
% guilty") is that the two conditions are *truly identical* -- any difference
% you measured is pure chance, nothing more. The p-value asks: *if that
% starting assumption were true, how surprising is the gap I actually
% measured?* A small p-value (below the significance level $\alpha = 0.05$)
% means "very surprising -- hard to explain as luck," so the difference is
% called *significant*.
%
% *This is the single most common misreading of statistics, so read it
% twice:* a p-value above 0.05 does *not* prove the two conditions are equal.
% It means the evidence was not strong enough, with only this many
% participants, to rule out luck. "Not significant" = "not demonstrated",
% never "proven the same".
%
% *In your data:* task time, Clutch vs Joystick, gives p = 0.380 -- not
% significant. That does not mean the two modes take equally long; it means
% 12 participants were not enough to separate them on this particular metric.
% Compare that with safety, Clutch vs Joystick: p < 0.001 -- here the gap is
% unmistakably real, Joystick is clearly safer.

%% 2.2 "How big is it?" -- effect size
% The p-value only says "probably real"; the effect size says "how much".
% Four different rulers are used here, one per kind of test -- the *bands*
% below (small / medium / large) are the standard convention for judging
% each one at a glance:
%
% * *Cohen's dz* (two-condition comparisons, e.g. Clutch vs Joystick): take
% every participant's own personal gap (their Clutch time minus their
% Joystick time), then divide the *average* of those personal gaps by how
% much the gaps *vary* from person to person. In plain words: "the average
% gap, measured in units of how much people normally differ from each
% other." $d_z = \bar{d} / s_d$. Bands: 0.2 small, 0.5 medium, 0.8 large.
% * *Rank-biserial r* (the safer, rank-only version of the above, used by
% the Wilcoxon test): simply "what fraction of participants leaned one way
% versus the other," rescaled so $r=+1$ means literally everyone went the
% same direction and $r=0$ means it was a coin flip who did better under
% each condition. $r = (W^+ - W^-)/(W^+ + W^-)$, from the rank sums $W^\pm$
% of the positive/negative personal gaps. Bands: 0.1 small, 0.3 medium, 0.5
% large.
% * *Partial eta squared* $\eta_p^2$ (three-or-more-condition ANOVA, e.g.
% F/B/FB): "what share of the ups and downs between conditions is explained
% by this factor, once person-to-person differences are removed." $\eta_p^2
% = SS_{effect} / (SS_{effect} + SS_{error})$. Bands: 0.01 small, 0.06
% medium, 0.14 large.
% * *Kendall's W* (agreement across participants, Sections 10-11): imagine
% giving 12 people the same 3 ice-cream flavours and asking each to rank
% favourite to least favourite. $W=1$ means everyone produced the identical
% ranking; $W=0$ means their rankings have nothing to do with each other.
% $W = \chi^2_F / (n(k-1))$. Bands: 0.1 weak, 0.3 moderate, 0.5 strong.
%
% *In your data:* safety, Clutch vs Joystick, |dz| = 1.74 -- a huge effect,
% Joystick is not just "significantly" safer, it is *substantially* safer.
% Assistance (F/B/FB) on time-effectiveness: partial $\eta_p^2 = 0.70$ --
% assistance level explains 70% of the trial-to-trial swing in that score.

%% 2.3 The confidence interval -- a weather forecast for the gap
% "70 degF +/- 5 degF" does not claim the temperature is exactly 70 -- it
% says you are fairly confident the truth sits somewhere between 65 and 75.
% The 95% confidence interval (CI) works the same way for the true gap
% between two conditions: it is the plausible range, not a single number. A
% CI that does not include 0 is the exact same conclusion as "significant",
% just expressed as a range instead of a single p-value -- and it is more
% informative, because it also tells you how big the gap plausibly is, not
% only whether it is nonzero.
%
% With only 12-24 participants there is no reliable formula for this range,
% so the report uses *bootstrapping*: it treats your participants as a
% stand-in for "everyone who could have done this study", and repeatedly
% (5000 times) redraws a pretend group of the same size *from your own data,
% with replacement* (so the same participant can be picked more than once in
% one redraw). Each pretend redraw gives a slightly different average gap;
% the middle 95% of those 5000 redraws is the confidence interval. It is a
% simulation of "how much would my answer wobble if I had tested a slightly
% different group of the same size."

%% 2.4 Two calculators, one trusted answer
% Every comparison is run through *two different calculators* at once:
%
% * a *parametric* one (t-test, ANOVA) -- more powerful, but it assumes the
% differences between conditions are roughly bell-curve shaped;
% * a *non-parametric* one (Wilcoxon, Friedman) -- a bit less powerful, but
% it only looks at who ranked above whom, so it cannot be thrown off by one
% unusually large or small trial (an outlier) or by a non-bell-curve shape.
%
% A quick automatic check on your actual data (a normality test on the
% differences, and for ANOVA a sphericity test) picks which of the two is
% *recommended* for that specific metric, and its p-value becomes the
% headline number. Both results are always kept in the tables so you can see
% them agree (reassuring) or disagree (a sign the result is fragile and
% should be read with extra caution).

%% 2.5 Testing many things at once -- the lottery-ticket problem
% If you buy 100 lottery tickets, one of them might "win" purely by luck --
% that does not mean lottery tickets work. Testing many metrics has exactly
% the same problem: test ~30 things and, on average, one or two will look
% "significant" by pure chance even if nothing real is happening anywhere.
%
% The *Holm-Bonferroni correction* fixes this by automatically raising the
% bar for significance based on how many tests are being run together in one
% batch: sort the p-values from smallest to largest, multiply the smallest
% by the number of tests $m$, the next by $m-1$, and so on, keeping the
% sequence non-decreasing. This is applied twice: (1) *inside* one test, over
% its own pairwise comparisons (3 pairs for F/B/FB, 15 pairs for the six
% cells); (2) *across* every metric of one family, within one question. Both
% the original ("raw") and the corrected ("Holm") p-value are always shown
% side by side, so you can see the correction happening rather than take it
% on faith. The one exception: the *family scores* (Section 4) are each
% reported on their own, uncorrected, because there is only one test per
% family and nothing to correct for.

%% 2.6 What "best" means here
% Every metric has a built-in direction: lower task time is better, higher
% clearance from obstacles is better, and so on (Section 3 states the
% direction of every metric explicitly). "Best" in this report always means
% "better in that direction, and the difference survived the checks above" --
% never a vague impression. A handful of metrics have no direction at all
% (mean speed, autonomy authority share) -- those are shown purely for
% context, never as a win or a loss.

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
% The collision barrier publishes at every tick, *per arm*, its own clearance
% $h_a(t)$: the SoftMin over that arm's active collision pairs of the signed
% distance to every other body (the other arm, the objects, the rack, the
% table). A cylinder that arm is carrying is fused to its gripper and excluded,
% and the pair being deliberately grasped is bypassed, so $h_a$ measures
% distance to *obstacles*, not to the payload. $h_a$ is a smooth lower bound of
% the true nearest distance (conservative by up to a couple of centimetres when
% several bodies are equally close), is capped at the 0.15 m sensing range, and
% samples inside the autonomous-grasp window are excluded. The controller's
% raw scalar minimum over all pairs is *not* used: it reads $-3.5$ cm whenever
% a cylinder is held, because the payload overlaps the gripper's own envelope.
%
% * *Minimum clearance (worst hand)* $\min_a \min_t h_a(t)$ over the
% teleoperated samples. Higher is better.
% * *Mean clearance* $\overline{h_a(t)}$, weighted over the two arms by the time
% each was active. Higher is better.
% * *Near-miss time fraction*: share of teleoperated samples with $h_a(t) <
% 0.05$ m, weighted over arms. Lower is better.
% * *Near-miss episodes*: number of separate dips of $h_a(t)$ below 0.05 m
% (rising edges of the indicator), summed over both arms. Lower is better.
% * *Safety filter active*: fraction of samples in which the Lagrange
% multiplier $\lambda$ of the collision barrier constraint exceeds 1, i.e. the
% safety filter is actively altering the commanded motion. Lower is better:
% the operator kept the robot away from the barrier by themselves. Its
% companion *safety filter active time* (the same in seconds) is a diagnostic:
% a fraction rises when the trial gets shorter even if the barrier time does not.
% * *Typical barrier push*: the median of $\lambda$ over the samples where it
% exceeded 1. The plain mean is heavy-tailed -- a hard barrier fight spikes
% $\lambda$ by three to six orders of magnitude for seconds and then owns the
% average -- so the mean is reported only as a diagnostic.

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
% robot became sure of what the operator wanted. In this data set it saturates
% above 0.99 in every trial, so it separates nothing.
% * *Mean intent confidence* $\overline{\max_k P_k(t)}$: the time-average of
% the leading goal's probability over the trial, which also counts how long the
% estimator stayed unsure after every reset. Higher is better.
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

%% 5. The tests, in plain language
% Every test below follows the same two-question recipe from Section 2 ("is
% it real", "how big is it") -- what differs from test to test is only *what
% is being compared*. All of them work on one value per participant per
% condition (that participant's own trials in that condition, averaged), so
% $n$ below is always the number of participants, never the number of
% trials.

%% 5.1 Two conditions -- comparing each person against themselves
% *Used for:* Clutch vs Joystick (Q1), B vs FB (part of Q2), rack vs shield.
%
% *In plain words:* every participant did both conditions, so instead of
% comparing the Clutch group to the Joystick group (they are the same 12
% people!), the test looks at each person's own personal gap -- "how much
% longer did *I* take in Clutch than in Joystick" -- and then asks whether
% those 12 personal gaps mostly lean one way, or whether they are scattered
% randomly around zero. This is exactly what makes it a *paired* test, and
% why it needs far fewer people than comparing two separate groups would.
%
% *The two calculators:* the paired t-test averages the personal gaps and
% compares that average to how much the gaps vary from person to person
% ($t = \bar d / (s_d/\sqrt n)$, $d_i = x_i - y_i$). The Wilcoxon signed-rank
% test instead ranks the 12 gaps by size (ignoring sign) and checks whether
% the ranks of the "Clutch was longer" gaps and the "Joystick was longer"
% gaps are balanced or lopsided (exact for $n \le 15$). A simple *sign test*
% is also reported as the plainest possible summary: literally just "how
% many of the 12 people went each way."

%% 5.2 Three conditions -- extending the same idea to three flavours
% *Used for:* F vs B vs FB (Q2).
%
% *In plain words:* the same "each person is their own comparison" logic,
% now with three conditions instead of two. First an *omnibus* test asks the
% broad question "do these three conditions differ from each other at all?"
% -- only if the answer is yes do we go looking for *which specific pairs*
% differ (F vs B? F vs FB? B vs FB?), each of those three pairwise checks
% Holm-corrected so that having three chances to find a difference does not
% by itself inflate the odds of a false alarm (Section 2.5).
%
% *The two calculators:* the repeated-measures ANOVA splits each
% participant's three values into "how this person differs from everyone
% else" and "how this condition differs from the others", $x_{ij} = \mu +
% \pi_i + \tau_j + \epsilon_{ij}$, and asks whether the condition part is
% large compared to the leftover noise, $F = MS_{condition}/MS_{error}$. It
% assumes the three pairwise gaps are similarly variable (*sphericity*,
% checked with Mauchly's test; when violated, the degrees of freedom are
% shrunk with the Greenhouse-Geisser correction so the test does not become
% falsely confident). The Friedman test instead ranks each person's own
% three values 1st/2nd/3rd and checks whether those rankings are consistent
% across people rather than random.

%% 5.3 Six cells -- does mode matter, does assistance matter, do they interact?
% *Used for:* the full mode x assistance grid (Q3).
%
% *In plain words:* this asks three separate questions from one model: (1)
% averaging over assistance, does *mode* matter? (2) averaging over mode,
% does *assistance* matter? (3) does the effect of assistance *depend on*
% which mode you are in -- i.e. does blending help more in Joystick than in
% Clutch, or the same in both? That third question is the *interaction*, and
% it is the one that a simple "compare each factor separately" analysis
% would miss entirely. Each of the three questions gets its own
% Greenhouse-Geisser-corrected p-value and effect size. When the interaction
% is significant, the report also breaks it down further ("assistance within
% Clutch only", "assistance within Joystick only") because the single
% overall "assistance effect" number would be misleading on its own. A
% Friedman test across all six cells together answers the plainest version
% of the question, "do the six cells differ at all", and the 15 possible
% pairwise comparisons (Holm-corrected) say exactly which cells differ from
% which.

%% 5.4 Consistency -- who agrees with themselves more?
% *Used for:* Q4 (mode) and Q5 (assistance).
%
% *In plain words:* every test so far asked "which condition has the better
% *average*?" This one asks a completely different question: "which
% condition gives more *similar* results from one participant to the next?"
% A condition where everybody scores about the same is more *predictable* --
% useful to know even when the average is a tie. This is measured as the
% spread (standard deviation) of the 12 participants' own averages under
% each condition; a shorter spread means more agreement between people.
%
% Two conditions' spreads are compared with the *Pitman-Morgan test*: a
% clever trick where, instead of comparing the two spreads directly, you
% look at the *sum* and the *difference* of each person's two values; if the
% two original conditions truly have equal spread, that sum and that
% difference will be statistically unrelated (correlation exactly 0), so the
% test simply checks whether they are. *Kendall's W* (Section 2.2) answers a
% related but different question on the same data: not "how spread out are
% the numbers", but "do participants at least *agree on which condition is
% better*, even if by different margins."
%
% *In your data:* task time is significantly more consistent in Joystick
% than in Clutch (participant-to-participant spread 33.7 vs 63.5,
% Pitman-Morgan Holm p = 0.005) -- even though Section 5.1 found no
% significant *average* time difference between the two modes. Joystick
% gives more predictable task times, without necessarily being faster.

%% 5.5 Learning and order -- did people get better, and does the order matter?
% *Used for:* Q6.
%
% *In plain words:* two separate concerns share this section. First,
% *learning*: across the six slots of the experiment (slot 1 = the very
% first condition a participant met), does performance drift up or down as
% people get more practice? For each participant a straight line is fitted
% through their own six slot-averages, giving one *slope* per person (their
% personal rate of improvement or decline); those 12 personal slopes are
% then tested against zero exactly like the paired gaps in Section 5.1 --
% "do the slopes mostly lean toward improvement, or are they scattered
% around flat."
%
% Second, *order bias*: because every participant did all three cells of one
% control mode before switching to the other, anyone who happened to meet
% Joystick second had three extra conditions of practice behind them before
% ever trying it -- which could quietly inflate an apparent "Joystick is
% better" result that is really just "practice is better". To check this,
% participants are split into two groups by which mode they met first, and
% the Joystick-minus-Clutch gap of the "Clutch-first" group is compared with
% that of the "Joystick-first" group (Mann-Whitney rank-sum test, the
% two-independent-groups cousin of the Wilcoxon test from Section 5.1, used
% here because these two groups are made of *different* people, not paired).
% If the two groups tell a different story, part of the mode comparison in
% Q1 is practice, not the mode itself -- the report flags this explicitly
% whenever it happens.
%
% *In your data:* task time shows no significant slope across the session
% (p = 0.077, borderline) but a significant *block* effect -- the three
% slots of the second-met mode are faster than the first three (p = 0.045)
% -- and the mode-order check comes back clean (p = 0.093, no significant
% dependence on which mode came first), so the Q1 mode comparison for task
% time is not contaminated by practice.

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

%% 13. Metric dashboards
% One figure per metric, four stacked panels: Q1 mode, Q2 assistance, Q3
% cells, Q6 trend. Grey lines are participants, coloured markers are means
% with 95% CI, brackets mark Holm-significant pairs. Larger versions of the
% same figures are in figures/metric_<name>.png in the results folder.
headlineMetrics = ["duration_s" "qdot_cmd_rms" "safety_min_dist_m" "ee_sparc" "belief_mean_prob" "agreement_mean_cos" "composite"];
metricItems = items(~items.is_family | items.name == "composite", :);
if ~SHOW_ALL_METRIC_FIGURES
    metricItems = metricItems(ismember(metricItems.name, headlineMetrics), :);
end
for k = 1:height(metricItems)
    fig_metric_dashboard(trial, table2struct(metricItems(k, :)), S, 'compact', true);
end

%% 14. Assumptions, limitations, and how to judge a result
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
% * *Task outcome is not analysed.* The experimenter's success / incident
% notes were not recorded consistently, and the study targets the assistance
% toward the goal poses rather than placement precision, so no outcome
% metric is used. Intent confidence is almost always reached, so the
% time-to-confidence is the informative belief metric.
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

%% 15. Appendix: full results table and provenance
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
