function spec = study_metric_spec()
%STUDY_METRIC_SPEC Single source of truth for every per-trial metric of the study.
%   SPEC = STUDY_METRIC_SPEC() returns a table with one row per metric of the
%   manifest (right_<name>/left_<name> columns written by export_to_matlab.py)
%   plus a few derived ones. Every downstream script (loader, tests, figures,
%   report) reads direction / family / scope from here, never hard-codes them.
%
%   Columns
%     name            manifest suffix (right_<name>) or derived name
%     label           plain-language name used in figures and tables
%     unit            axis unit
%     dir             +1 higher is better, -1 lower is better, 0 no valence
%     family          metric family used for grouping and composite scores;
%                     "" = diagnostic only (reported, never in a composite)
%     combine         how the right_/left_ columns fold into one trial value:
%                     shared  identical in both columns (trial-level quantity)
%                     sum     extensive per-arm quantity, both arms add
%                     wmean   intensive per-arm quantity, weighted by the time
%                             each arm was the active one
%                     max     worst case over the two arms (lower is better)
%                     min     worst case over the two arms (higher is better)
%                     derived computed from other columns (see load_study_table)
%     cell_scope      "all" | "clutch_only" | "blend_only": cells where the
%                     metric is physically defined; elsewhere it is NaN
%     mode_comparable false when CLUTCH and JOYSTICK values have a different
%                     physical origin (haptic force, clutch button): such a
%                     metric is compared only within one control mode
%     cv_safe         true when the coefficient of variation is meaningful
%                     (positive, not near zero, not a signed score)
%     description     one-sentence definition (the full math is in the report)

rows = {
% name                        label                              unit    dir family                 combine  scope         modecmp cvsafe description
 "duration_s"                 "Task time"                        "s"     -1  "time_effectiveness"   "shared" "all"         true   true  "Bag time span from the first to the last recorded message of the trial."
 "teleop_time_s"              "Time under human control"         "s"     -1  "time_effectiveness"   "derived" "all"        true   true  "Task time minus the time spent in autonomous grasp/release phases."
 "autonomy_grasp_time_s"      "Time in autonomous grasp phases"  "s"     -1  "time_effectiveness"   "shared" "all"         true   true  "Seconds during which grasp_active was true (robot-driven approach, close, lift, release)."
 "ee_path_len_m"              "Hand path length (both arms)"     "m"     -1  "time_effectiveness"   "sum"    "all"         true   true  "Arc length travelled by the two end effectors, summed."
 "ee_path_efficiency"         "Path efficiency"                  "ratio" +1  "time_effectiveness"   "derived" "all"        true   true  "Straight-line displacement divided by path length (ratio of arm sums)."
 "ee_speed_mean_mps"          "Mean hand speed"                  "m/s"    0  "time_effectiveness"   "wmean"  "all"         true   true  "Mean end-effector linear speed of the active arm."
 "qdot_cmd_rms"               "Commanded joint-rate RMS"         "rad/s" -1  "human_effort"         "wmean"  "all"         true   true  "Root-mean-square of the QP joint velocity command over the 7 arm joints."
 "force_mean_N"               "Mean haptic force"                "N"     -1  "human_effort"         "shared" "all"         false  true  "Mean magnitude of the wrench rendered on the handle (tether in CLUTCH, centring spring in JOYSTICK)."
 "force_peak_N"               "Peak haptic force"                "N"     -1  "human_effort"         "shared" "all"         false  true  "Maximum rendered force magnitude."
 "force_impulse_Ns"           "Haptic force impulse"             "N.s"   -1  "human_effort"         "shared" "all"         false  true  "Time integral of the rendered force magnitude."
 "clutch_presses"             "Clutch presses"                   "count" -1  "human_effort"         "shared" "clutch_only" false  true  "Number of rising edges of the clutch button (CLUTCH mode only)."
 "clutch_duty_frac"           "Clutch duty"                      "frac"  -1  "human_effort"         "shared" "clutch_only" false  true  "Fraction of samples with the clutch button held (CLUTCH mode only)."
 "safety_min_dist_m"          "Minimum clearance (worst hand)"   "m"     +1  "safety"               "min"    "all"         true   true  "Closest either arm's own clearance (barrier SoftMin, carried cylinder excluded) came to an obstacle while the operator was driving."
 "safety_mean_dist_m"         "Mean clearance"                   "m"     +1  "safety"               "wmean"  "all"         true   true  "Time-average of the arm's clearance while the operator was driving, capped at the 0.15 m sensing range."
 "safety_nearmiss_frac"       "Near-miss time fraction"          "frac"  -1  "safety"               "wmean"  "all"         true   false "Fraction of teleoperated samples with the arm's clearance below 0.05 m."
 "safety_nearmiss_episodes"   "Near-miss episodes (both arms)"   "count" -1  "safety"               "sum"    "all"         true   false "Number of separate dips of either arm's clearance below 0.05 m."
 "cbf_active_frac"            "Safety filter active"             "frac"  -1  "safety"               "wmean"  "all"         true   false "Fraction of operator-driven samples (autonomous grasp phases excluded) where the collision barrier multiplier is non-zero (above the 1e-3 dual tolerance)."
 "cbf_lambda_active_median"   "Typical barrier push"             "-"     -1  "safety"               "wmean"  "all"         true   true  "Median collision-barrier multiplier over the samples where it was active (robust to spikes)."
 "ee_sparc"                   "Smoothness (SPARC)"               "-"     +1  "motion_quality"       "wmean"  "all"         true   false "Mean spectral arc length over the active-driving segments (clutch released, hand above stillness, >= 0.5 s); negative, closer to 0 is smoother."
 "ee_sparc_whole"             "Smoothness, whole profile"        "-"     +1  ""                     "wmean"  "all"         true   false "Spectral arc length of the whole hand speed profile, holds included; superseded by the segmented value, kept for the before/after check."
 "ee_sparc_nseg"              "Driving segments scored"          "count"  0  ""                     "wmean"  "all"         true   false "Number of active-driving segments that entered the segmented smoothness."
 "slack_mean"                 "Mean tracking slack"              "-"     -1  "motion_quality"       "wmean"  "all"         true   true  "Mean relaxation of the tracking constraint (how far the robot deviated from the reference)."
 "qdot_cmd_max"               "Peak commanded joint rate"        "rad/s" -1  "motion_quality"       "max"    "all"         true   true  "Largest absolute joint velocity command."
 "belief_max_prob"            "Peak intent confidence"           "prob"  +1  "intent_understanding" "shared" "all"         true   true  "Highest probability assigned to any goal during the trial."
 "belief_mean_prob"           "Mean intent confidence"           "prob"  +1  "intent_understanding" "shared" "all"         true   true  "Time-average of the probability of the currently most likely goal."
 "belief_time_to_conf_s"      "Time to confident intent"         "s"     -1  "intent_understanding" "shared" "all"         true   true  "First time at which the most likely goal reached probability 0.80 (NaN if never)."
 "belief_confident_ever"      "Intent ever confident"            "0/1"   +1  "intent_understanding" "derived" "all"        true   false "1 if the belief reached 0.80 at least once, else 0."
 "agreement_mean_cos"         "User-autonomy agreement"          "cos"   +1  "assistance_quality"   "shared" "blend_only"  true   false "Mean over active driving of the per-channel cosine between user and policy twist (linear always, angular when driven), as the controller computes its alignment."
 "agreement_lin_mean_cos"     "Agreement, linear channel"        "cos"   +1  ""                     "shared" "blend_only"  true   false "Mean cosine of the linear channel alone over active driving."
 "agreement_ang_mean_cos"     "Agreement, angular channel"       "cos"   +1  ""                     "shared" "blend_only"  true   false "Mean cosine of the angular channel over the driving samples where it was above stillness."
 "agreement_mean_cos6d"       "Agreement, 6-D cosine (legacy)"   "cos"   +1  ""                     "shared" "blend_only"  true   false "Cosine of the full six-dimensional twists, m/s and rad/s mixed; superseded, kept for the before/after check."
 "alpha_mean"                 "Mean autonomy authority"          "-"      0  "assistance_quality"   "shared" "blend_only"  true   false "Mean blending weight alpha (0 = user only, 1 = policy only)."
 "alpha_autonomy_frac"        "Autonomy-led time fraction"       "frac"   0  "assistance_quality"   "shared" "blend_only"  true   false "Fraction of active-driving samples with alpha above 0.5."
 "alpha_autonomy_frac_total"  "Autonomy-led, whole trial"        "frac"   0  ""                     "shared" "blend_only"  true   false "Fraction of all samples with alpha above 0.5, holds and grasps included; superseded, kept for the before/after check."
 "user_active_frac"           "User actively driving"            "frac"   0  "assistance_quality"   "shared" "blend_only"  true   false "Fraction of samples in the active-driving set: operator in control, clutch released, hand above stillness."
 "intervention_mean_mps"      "Command alteration (mean)"        "m/s"    0  ""                     "shared" "blend_only"  true   false "Mean ||v_blend - v_user|| while the operator drives: how far arbitration moved the command away from what was asked. The transparency measure of the shared-control literature; no direction is assigned because altering the command is the point of blending, not a defect."
 "intervention_peak_mps"      "Command alteration (peak)"        "m/s"    0  ""                     "shared" "blend_only"  true   false "Largest single-sample ||v_blend - v_user|| while the operator drives."
 "safety_min_dist_graspincl_m" "Raw min. distance (legacy)"      "m"      0  ""                     "shared" "all"         true   false "Raw scalar over all pairs including a carried cylinder; reads -3.5 cm whenever a cylinder is held, kept only for reference."
 "cbf_active_s"               "Safety filter active time"        "s"     -1  ""                     "derived" "all"        true   true  "Seconds of operator-driven time with the collision barrier multiplier non-zero (active fraction x time under human control)."
 "safety_nearmiss_s"          "Near-miss time"                   "s"     -1  ""                     "derived" "all"        true   true  "Seconds of operator-driven time with the arm's clearance below 0.05 m (near-miss fraction x time under human control)."
 "cbf_lambda_mean"            "Mean barrier multiplier"          "-"     -1  ""                     "wmean"  "all"         true   true  "Mean Lagrange multiplier of the collision barrier; heavy-tailed, dominated by rare spikes."
 "cbf_lambda_peak"            "Peak barrier multiplier"          "-"     -1  ""                     "max"    "all"         true   true  "Largest collision-barrier multiplier."
 "slack_peak"                 "Peak tracking slack"              "-"     -1  ""                     "max"    "all"         true   true  "Largest tracking-constraint relaxation."
 "qdot_meas_rms"              "Measured joint-rate RMS"          "rad/s" -1  ""                     "wmean"  "all"         true   true  "RMS of the measured joint velocity."
 "qdot_meas_max"              "Peak measured joint rate"         "rad/s" -1  ""                     "max"    "all"         true   true  "Largest measured joint velocity."
 "ee_speed_max_mps"           "Peak hand speed"                  "m/s"    0  ""                     "max"    "all"         true   true  "Maximum end-effector speed."
 "ee_straight_len_m"          "Straight-line displacement"       "m"      0  ""                     "sum"    "all"         true   true  "Distance between the first and last hand position, summed over arms."
 "autonomy_grasp_frac"        "Autonomous grasp fraction"        "frac"   0  ""                     "shared" "all"         true   false "Fraction of samples in autonomous grasp phases."
 "loop_freq_mean_hz"          "Controller loop rate"             "Hz"     0  ""                     "shared" "all"         true   true  "Self-reported control loop frequency (health check, not performance)."
};

spec = cell2table(rows, 'VariableNames', {'name','label','unit','dir','family', ...
    'combine','cell_scope','mode_comparable','cv_safe','description'});
spec.name = string(spec.name);       spec.label = string(spec.label);
spec.unit = string(spec.unit);       spec.family = string(spec.family);
spec.combine = string(spec.combine); spec.cell_scope = string(spec.cell_scope);
spec.description = string(spec.description);
end
