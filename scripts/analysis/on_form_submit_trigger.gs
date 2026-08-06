/**
 * on_form_submit_trigger.gs
 *
 * Attach this to the SPREADSHEET linked to the post-trial questionnaire
 * (not to the Form itself). Runs automatically on every new submission:
 *   - appends a "trial_key" column (Participant_World_Condition), matching
 *     study_config.bag_folder_name()'s <world_shortcut>_<cell_code>
 *     convention, so a subjective row can be joined against metrics.json
 *     without a manual formula;
 *   - appends a "duplicate?" column flagging if this exact
 *     Participant+World+Condition combination was already submitted before
 *     (catches an accidental re-submit or a Latin-square bookkeeping slip
 *     without blocking the submission itself).
 *
 * HOW TO INSTALL (one-time, ~20 seconds):
 *   1. Open the RESPONSE SPREADSHEET (not the Form) -- Extensions > Apps Script.
 *   2. Paste this file in, save.
 *   3. Run installTrigger() once from the editor (top toolbar function
 *      dropdown -> installTrigger -> Run). Approve the permission prompt.
 *   4. Done -- onFormSubmit now fires automatically on every future response.
 *      (It does NOT retroactively fill in rows submitted before you installed it.)
 */

function installTrigger() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  // Remove any previous copy of this trigger first, so re-running this
  // installer never creates duplicate triggers on the same sheet.
  ScriptApp.getProjectTriggers().forEach(function(t) {
    if (t.getHandlerFunction() === 'onFormSubmit') ScriptApp.deleteTrigger(t);
  });
  ScriptApp.newTrigger('onFormSubmit')
      .forSpreadsheet(ss)
      .onFormSubmit()
      .create();
  Logger.log('Trigger installed on: ' + ss.getUrl());
}

function onFormSubmit(e) {
  var sheet = e.range.getSheet();
  var headers = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0];
  var row = e.range.getRow();

  var col = {};
  headers.forEach(function(h, i) { col[h] = i + 1; });

  var participant = sheet.getRange(row, col['Participant ID']).getValue();
  var world = sheet.getRange(row, col['World']).getValue();
  var condition = sheet.getRange(row, col['Condition']).getValue();
  var trialKey = participant + '_' + world + '_' + condition;

  // Ensure the two helper columns exist (created once, on first submission).
  var keyCol = headers.indexOf('trial_key') + 1;
  if (!keyCol) {
    keyCol = sheet.getLastColumn() + 1;
    sheet.getRange(1, keyCol).setValue('trial_key');
  }
  var dupCol = headers.indexOf('duplicate?') + 1;
  if (!dupCol) {
    dupCol = sheet.getLastColumn() + 1;
    sheet.getRange(1, dupCol).setValue('duplicate?');
  }

  // Duplicate check: same trial_key already present in an earlier row.
  var existingKeys = sheet.getRange(2, keyCol, Math.max(row - 2, 0), 1).getValues()
      .map(function(r) { return r[0]; });
  var isDuplicate = existingKeys.indexOf(trialKey) !== -1;

  sheet.getRange(row, keyCol).setValue(trialKey);
  sheet.getRange(row, dupCol).setValue(isDuplicate ? 'YES -- check this trial' : '');
}
