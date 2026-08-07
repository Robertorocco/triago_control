/**
 * build_post_trial_form.gs
 *
 * Builds the post-condition questionnaire as a real Google Form, matching
 * post_trial_questionnaire.tex's trimmed item set (3 NASA-TLX + 3 HRI items).
 * Filled ONCE per condition, after both worlds have been tried under it --
 * 6 submissions per participant (the 6 non-tutorial cells; C/J are excluded).
 * No World field: since one submission already covers both worlds, asking
 * which world would be meaningless (and there'd be nothing to select for).
 *
 * HOW TO RUN (one-time setup, ~30 seconds):
 *   1. Go to https://script.google.com -> New project.
 *   2. Delete the placeholder code, paste this whole file in.
 *   3. Click Run (the play button) on buildPostTrialForm.
 *   4. First run asks for permission (it needs to create a Form in your
 *      Drive) -- Google will warn "unverified app" since this is your own
 *      script, not a public one; click Advanced -> Go to project (unsafe).
 *   5. Check the Execution log (View > Logs, or Ctrl+Enter) for two links:
 *      the EDIT url (to look at/tweak the form) and the LIVE url (what you
 *      actually hand to participants / open at the operator station).
 *   6. In the form's Responses tab, click the green Sheets icon once to
 *      create the linked spreadsheet every submission will land in.
 *
 * You only need to run this once total -- it creates ONE form reused for
 * every condition of every participant. If you already built the OLDER
 * per-trial version of this form (with a World question), don't re-run this
 * and get a second form/URL -- just delete the "World" question from the
 * existing form directly in the Forms UI; everything else is unchanged.
 */

var CONDITION_CODES = ['CF', 'CB', 'CFB', 'JF', 'JB', 'JFB']; // C and J excluded: tutorial-only

function buildPostTrialForm() {
  var form = FormApp.create('Post-Trial Questionnaire');
  form.setDescription(
    'Complete once per condition, after both worlds have been tried under ' +
    'it. There are no right answers -- answer how the condition actually felt.'
  );

  // One form reused for all 6 conditions/participant -- disable both settings
  // that would otherwise block a participant from submitting more than once.
  form.setCollectEmail(false);
  form.setLimitOneResponsePerUser(false);
  form.setShuffleQuestions(false);

  // --- Identification (structured, not free text, so nothing here can typo) ---
  form.addTextItem()
      .setTitle('Participant ID')
      .setRequired(true);

  form.addListItem()
      .setTitle('Condition')
      .setChoiceValues(CONDITION_CODES)
      .setRequired(true);
  // No Trial# field: Condition alone already identifies each of the 6
  // submissions for a participant. No Date/Time field: Forms timestamps
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
      .setTitle('Notes -- anything notable about this condition? (optional)')
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
