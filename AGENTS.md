# Working preferences

## Automatic cleanup after tasks

- The user authorizes routine cleanup of disposable files created or downloaded
  for a task once verification is finished, without asking again. Include
  one-off validation scripts, scratch files, completed installation archives,
  abandoned partial downloads, and redundant task copies when no longer needed.
- Verify exact targets and dependencies before removal. Limit cleanup to known
  task artifacts; age or absence from the current workflow alone does not
  establish that a file is unused.
- When the user retires a workflow, also remove models downloaded exclusively
  for it after checking for shared dependencies. Preserve active or shared
  models, required tools and environments, source files, reference inputs, and
  user outputs unless their removal is authorized.
- Prefer the Recycle Bin when practical. Do not empty the Recycle Bin or
  permanently remove unrelated files as part of routine cleanup.
- Briefly report material cleanup, approximate size when useful, and whether
  files are recoverable. Perform cleanup as part of task completion; this
  preference does not create a background schedule.

# Local-first project workflow

- Develop, review, and test in this checkout by default. Local Git status, diffs,
  branches, and commits do not need GitHub.
- Do not fetch, pull, push, open pull requests, query GitHub APIs, poll hosted
  checks, or dispatch workflows as a routine part of local work. Use remote
  operations only when the user explicitly requests GitHub sync, backup, review,
  or another task that requires them. Batch related changes for occasional sync.
- Run `python scripts/check_project.py` for syntax checks and the unit suite.
  Use `--verbose` when detailed test names are useful. For compatibility changes,
  also use the locally installed Python 3.10 and 3.11 interpreters if available.
  Do not install packages or download models automatically as part of validation.
- Keep the tracked Tests workflow manual-only unless the user asks to restore
  automatic CI. Local Windows checks do not prove Linux compatibility; the
  optional full hosted matrix covers both platforms.
- Leave GitHub security scanning and branch protections unchanged unless the
  user explicitly requests changes to those settings. Repository-hosted security
  automation is separate from the tracked Tests workflow.
- Preserve unrelated local changes and never include credentials, `.env` files,
  recordings, or local backup bundles in a GitHub sync.
