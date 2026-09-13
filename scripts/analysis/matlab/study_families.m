function fam = study_families()
%STUDY_FAMILIES Ordered catalogue of metric families with a plain-language meaning.
%   FAM = STUDY_FAMILIES() returns a table (key, label, meaning). The order is
%   the order used in every report section and overview figure.

rows = {
 "time_effectiveness"   "Time & effectiveness"   "How quickly and directly the task was completed."
 "human_effort"         "Human effort"           "How much the operator had to work: joint-rate demand, haptic load, clutch use."
 "safety"               "Safety"                 "How close the robot came to obstacles and how often the safety filter had to intervene."
 "motion_quality"       "Motion quality"         "How smooth and well tracked the robot motion was."
 "intent_understanding" "Intent understanding"   "How confidently and how early the robot inferred which object the operator wanted."
 "assistance_quality"   "Assistance quality"     "How well the autonomous policy agreed with the operator (blending cells only)."
};
fam = cell2table(rows, 'VariableNames', {'key','label','meaning'});
fam.key = string(fam.key); fam.label = string(fam.label); fam.meaning = string(fam.meaning);
end
