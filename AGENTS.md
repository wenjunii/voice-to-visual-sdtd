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
