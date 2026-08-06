/**
 * build_post_trial_form.gs
 *
 * Builds the post-trial questionnaire as a real Google Form, matching
 * post_trial_questionnaire.tex's trimmed item set (3 NASA-TLX + 3 HRI items),
 * with the identification fields redesigned for 12 repeated submissions per
 * participant (2 worlds x 6 conditions) instead of the paper form's blanks.
 *
 * HOW TO RUN (one-time setup, ~30 seconds):
 *   1. Go to https://script.google.com -> New project.
 *   2. Delete the placeholder code, paste this whole file in.
 *   3. EDIT THE WORLD_NAMES ARRAY BELOW to your actual 2 world names.
 *   4. Click Run (the play button) on buildPostTrialForm.
 *   5. First run asks for permission (it needs to create a Form in your
 *      Drive) -- Google will warn "unverified app" since this is your own
 *      script, not a public one; click Advanced -> Go to project (unsafe).
 *   6. Check the Execution log (View > Logs, or Ctrl+Enter) for two links:
 *      the EDIT url (to look at/tweak the form) and the LIVE url (what you
 *      actually hand to participants / open at the operator station).
 *   7. In the form's Responses tab, click the green Sheets icon once to
 *      create the linked spreadsheet every submission will land in.
 *
 * You only need to run this once total -- it creates ONE form reused for
 * every trial of every participant, per the "one form, not twelve" design.
 */

// ---- EDIT THESE TWO LINES to match your actual study -----------------------
var WORLD_NAMES = ['no_obstacle', 'REPLACE_WITH_SECOND_WORLD_NAME'];
var CONDITION_CODES = ['CF', 'CB', 'CFB', 'JF', 'JB', 'JFB']; // C and J excluded: tutorial-only
// -----------------------------------------------------------------------------

function buildPostTrialForm() {
  var form = FormApp.create('Post-Trial Questionnaire');
  form.setDescription(
    'Complete immediately after this trial, before the next simulation ' +
    'condition is prepared. There are no right answers -- answer how the ' +
    'trial actually felt.'
  );

  // One form reused for all 12 trials/participant -- disable both settings
  // that would otherwise block a participant from submitting more than once.
  form.setCollectEmail(false);
  form.setLimitOneResponsePerUser(false);
  form.setShuffleQuestions(false);

  // --- Identification (structured, not free text, so nothing here can typo) ---
  form.addTextItem()
      .setTitle('Participant ID')
      .setRequired(true);

  form.addListItem()
      .setTitle('World')
      .setChoiceValues(WORLD_NAMES)
      .setRequired(true);

  form.addListItem()
      .setTitle('Condition')
      .setChoiceValues(CONDITION_CODES)
      .setRequired(true);
  // No Trial# field: World x Condition already uniquely identifies each of
  // the 12 trials for a participant. No Date/Time field: Forms timestamps
  // every submission automatically in the linked Sheet.

  // --- Section A: Workload (trimmed NASA-TLX: Mental/Physical Demand, Performance) ---
  addSevenPointScale(form, '1. Mental Demand -- How mentally demanding was the task?',
                     'Very Low', 'Very High');
  addSevenPointScale(form, '2. Physical Demand -- How physically demanding was the task?',
                     'Very Low', 'Very High');
  addSevenPointScale(form, '3. Performance -- How successful were you in accomplishing the task?',
                     'Perfect', 'Failure');

  // --- Section B: Control, Trust & Comfort ---
  addSevenPointScale(form, "4. I felt in control of the robot's motion.",
                     'Strongly Disagree', 'Strongly Agree');
  addSevenPointScale(form, '5. I trusted the robot to do what I intended.',
                     'Strongly Disagree', 'Strongly Agree');
  addSevenPointScale(form, "6. The robot's motion felt smooth and comfortable.",
                     'Strongly Disagree', 'Strongly Agree');

  // --- Section C: Notes ---
  form.addParagraphTextItem()
      .setTitle('Notes -- anything notable about this trial? (optional)')
      .setRequired(false);

  Logger.log('EDIT URL (for you): ' + form.getEditUrl());
  Logger.log('LIVE URL (open this at the operator station): ' + form.getPublishedUrl());
}

function addSevenPointScale(form, title, lowLabel, highLabel) {
  form.addScaleItem()
      .setTitle(title)
      .setBounds(1, 7)
      .setLabels(lowLabel, highLabel)
      .setRequired(true);
}
