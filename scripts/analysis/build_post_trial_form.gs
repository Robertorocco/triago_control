/**
 * build_post_trial_form.gs
 *
 * Builds the whole-session questionnaire as a single Google Form with three
 * kinds of section, all reached through one landing-page router so the same
 * form can be resubmitted many times per participant:
 *   - Before You Begin   -- one-shot, filled once at the very start: age,
 *                           gender, dominant hand, prior teleop experience
 *   - Condition <code>   -- 6 sections (one per non-tutorial condition):
 *                           trimmed NASA-TLX + HRI items (post_trial_questionnaire.tex)
 *   - End of Session     -- one-shot, filled once after the 6th condition:
 *                           best strategy, ranking grids, an open comments field
 *
 * Each visit is exactly one submission: the landing-page question routes
 * straight to the matching section, and only that section is ever shown.
 *
 * HOW TO RUN (one-time setup, ~30 seconds):
 *   1. Go to https://script.google.com -> New project.
 *   2. Delete the placeholder code, paste this whole file in.
 *   3. Click Run (the play button) on buildPostTrialForm.
 *   4. First run asks for permission (it creates a Form in your Drive) --
 *      Google warns "unverified app"; click Advanced -> Go to project (unsafe).
 *   5. Check the Execution log (View > Logs, or Ctrl+Enter) for the EDIT url
 *      (tweak the form) and the LIVE url (hand to participants / operator station).
 *   6. In the form's Responses tab, click the green Sheets icon once to
 *      create the linked spreadsheet every submission will land in.
 *
 * NOTE: this always mints a brand-new Form/URL via FormApp.create; it never
 * edits an existing form in place, so re-running it creates a second, separate form.
 */

var CONDITION_CODES = ['CF', 'CB', 'CFB', 'JF', 'JB', 'JFB']; // C and J excluded: tutorial-only

// Participant-facing names -- internal codes never appear on the form itself.
var CONDITION_LABELS = {
  CF: 'CLUTCH FEEDBACK',
  CB: 'CLUTCH BLENDING',
  CFB: 'CLUTCH FEEDBACK+BLENDING',
  JF: 'JOYSTICK FEEDBACK',
  JB: 'JOYSTICK BLENDING',
  JFB: 'JOYSTICK FEEDBACK+BLENDING'
};

// Reminds the participant what they just tried, in control-mode/feedback/blending terms.
var CONDITION_DESCRIPTIONS = {
  CF: 'Clutch (position) control. You felt haptic force feedback; your commands were not blended with the robot\'s suggestions.',
  CB: 'Clutch (position) control. No force feedback; your commands were blended with the robot\'s suggestions.',
  CFB: 'Clutch (position) control. You felt haptic force feedback and your commands were blended with the robot\'s suggestions.',
  JF: 'Joystick (velocity) control. You felt haptic force feedback; your commands were not blended with the robot\'s suggestions.',
  JB: 'Joystick (velocity) control. No force feedback; your commands were blended with the robot\'s suggestions.',
  JFB: 'Joystick (velocity) control. You felt haptic force feedback and your commands were blended with the robot\'s suggestions.'
};

function buildPostTrialForm() {
  var form = FormApp.create('User Study Questionnaire');
  form.setDescription(
    'One form for the whole session. Use it once before your first ' +
    'condition, once after each of the 6 conditions, and once at the very ' +
    'end -- pick which on the first question below.'
  );

  // One form reused many times per participant -- disable both settings
  // that would otherwise block more than one submission per person.
  form.setCollectEmail(false);
  form.setLimitOneResponsePerUser(false);
  form.setShuffleQuestions(false);

  // --- Landing page: identity + router ---
  form.addTextItem()
      .setTitle('Participant ID')
      .setRequired(true);

  var stageItem = form.addMultipleChoiceItem()
      .setTitle('What are you submitting right now?')
      .setRequired(true);

  // --- Section 1: pre-experiment one-shot ---
  var prePage = form.addPageBreakItem()
      .setTitle('Before You Begin')
      .setHelpText('Fill this once, before your first condition.');
  form.addTextItem()
      .setTitle('Age of participant')
      .setRequired(false);
  form.addMultipleChoiceItem()
      .setTitle('To which gender identity do you identify?')
      .setChoiceValues(['Male', 'Female', 'Other', 'Prefer not to say'])
      .setRequired(true);
  form.addMultipleChoiceItem()
      .setTitle('Which is your dominant hand?')
      .setChoiceValues(['Right', 'Left', 'Ambidextrous / no strong preference'])
      .setRequired(true);
  form.addScaleItem()
      .setTitle('How would you rate your level of experience or knowledge in using teleoperation devices?')
      .setBounds(1, 5)
      .setLabels('No experience/knowledge at all', 'Expert')
      .setRequired(true);
  prePage.setGoToPage(FormApp.PageNavigationType.SUBMIT);

  // --- Section 2: condition picker ---
  var pickerPage = form.addPageBreakItem()
      .setTitle('Condition Report')
      .setHelpText('Which condition did you just finish?');
  // Closed-choice field, not free text: the condition code can't be
  // mistyped, so it stays a reliable join key against the trial data.
  var conditionItem = form.addMultipleChoiceItem()
      .setTitle('Condition just completed')
      .setRequired(true);

  // --- Sections 3-8: one equivalent block per condition ---
  var conditionPages = {};
  CONDITION_CODES.forEach(function(code) {
    var page = form.addPageBreakItem()
        .setTitle(CONDITION_LABELS[code])
        .setHelpText(CONDITION_DESCRIPTIONS[code]);
    addSevenPointScale(form, '1. Mental Demand -- How mentally demanding was the task?',
                       'Very Low', 'Very High');
    addSevenPointScale(form, '2. Physical Demand -- How physically demanding was the task?',
                       'Very Low', 'Very High');
    addSevenPointScale(form, '3. Performance -- How successful were you in accomplishing the task?',
                       'Perfect', 'Failure');
    addSevenPointScale(form, "4. I felt in control of the robot's motion.",
                       'Strongly Disagree', 'Strongly Agree');
    addSevenPointScale(form, '5. I trusted the robot to do what I intended.',
                       'Strongly Disagree', 'Strongly Agree');
    addSevenPointScale(form, "6. The robot's motion felt smooth and comfortable.",
                       'Strongly Disagree', 'Strongly Agree');
    form.addParagraphTextItem()
        .setTitle('Notes -- anything notable about this condition? (optional)')
        .setRequired(false);
    page.setGoToPage(FormApp.PageNavigationType.SUBMIT);
    conditionPages[code] = page;
  });

  // --- Section 9: end-of-experiment one-shot ---
  var postPage = form.addPageBreakItem()
      .setTitle('End of Session')
      .setHelpText('Fill this once, after your 6th and final condition.');
  form.addMultipleChoiceItem()
      .setTitle('Which teleoperation strategy was the best?')
      .setChoiceValues(['Clutch (Position)', 'Joystick (Velocity)'])
      .setRequired(true);
  addRankingGrid(form, 'Rank the three clutch-assisted teleoperation conditions you experienced today, from most to least preferred.',
                 [CONDITION_LABELS.CF, CONDITION_LABELS.CB, CONDITION_LABELS.CFB]);
  addRankingGrid(form, 'Rank the three joystick-assisted teleoperation conditions you experienced today, from most to least preferred.',
                 [CONDITION_LABELS.JF, CONDITION_LABELS.JB, CONDITION_LABELS.JFB]);
  form.addParagraphTextItem()
      .setTitle('Do you have any additional comments or observations regarding the teleoperation strategies you experienced today?')
      .setRequired(false);
  postPage.setGoToPage(FormApp.PageNavigationType.SUBMIT);

  // --- Wire the routing now that every destination page exists ---
  stageItem.setChoices([
    stageItem.createChoice('My first submission today (before starting)', prePage),
    stageItem.createChoice('A condition report', pickerPage),
    stageItem.createChoice('My last submission today (end of session)', postPage)
  ]);

  conditionItem.setChoices(CONDITION_CODES.map(function(code) {
    return conditionItem.createChoice(CONDITION_LABELS[code], conditionPages[code]);
  }));

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

// Multiple-choice grid with one response per column -- each rank position
// (1st/2nd/3rd) can only be assigned to a single condition, forcing a proper
// ranking instead of allowing ties.
function addRankingGrid(form, title, items) {
  var validation = FormApp.createGridValidation()
      .requireLimitOneResponsePerColumn()
      .build();
  form.addGridItem()
      .setTitle(title)
      .setRows(items)
      .setColumns(['1st (most preferred)', '2nd', '3rd'])
      .setValidation(validation)
      .setRequired(true);
}
