---
name: spark-github-pr
description: Use when pushing a branch or opening a PR to GitHub from this DGX Spark box (sky4star forks). Covers why the stored fine-grained PAT fails for git-over-HTTPS and SSH must be used instead, creating a clean worktree from the fork's main, targeting main (there is no master), and redacting secrets before committing. Triggers include "发 PR", "提交到 github", "push 分支", "fork", "Authentication failed for github".
---

# Push / open a PR from this box

## Git transport: use SSH, not the stored token

`gh` is logged in as **sky4star** with a fine-grained PAT. That token works for
the **API** (`gh api`, `gh repo view`) but **fails for git-over-HTTPS**:

```
remote: Invalid username or token. Password authentication is not supported for Git operations.
```

`gh auth setup-git` configures the helper but the PAT still fails for git. Use
the SSH key instead — it authenticates as sky4star:

```bash
ssh -o BatchMode=yes -T git@github.com     # => "Hi sky4star! ..."
git remote set-url <remote> git@github.com:sky4star/<repo>.git
GIT_SSH_COMMAND='ssh -o BatchMode=yes' git fetch <remote> main
```

## Repos / branches

- Existing forks: **`sky4star/sglang`**, **`sky4star/KDA-Pilot`** (forks of
  `sgl-project/sglang` and `BBuf/KDA-Pilot`). Both default to **`main`**; there
  is **no `master`** — if asked for "master", target `main` and say so.
- The user has read-only on the parents; always push to the fork.

## Clean branch without disturbing the working checkout

The local checkouts may carry unrelated dirty edits / submodules. Do not
branch/commit on them; make a worktree off the fork's main:

```bash
cd ~/sglang            # or ~/kda/KDA-Pilot
git remote add sky4star git@github.com:sky4star/sglang.git 2>/dev/null || true
GIT_SSH_COMMAND='ssh -o BatchMode=yes' git fetch sky4star main
git worktree add -b <topic-branch> /tmp/../<worktree> sky4star/main
```

Set identity per worktree (global is empty):
```bash
git config user.name sky4star
git config user.email 10058436+sky4star@users.noreply.github.com   # gh api user --jq .id
```

## Always redact secrets first

Before staging, scan and replace any API key / token. The serving API key tends
to be hardcoded in copied run scripts and e2e clients — parameterize it:

```bash
# scan for secrets; this must return nothing before commit
grep -rInE '[0-9a-f]{40,}|github_pat_|sk-[A-Za-z0-9]' <dir>

# shell:    APIKEY="${SGLANG_API_KEY:-REPLACE_WITH_API_KEY}"
# python:   DEFAULT_KEY = os.environ.get("SGLANG_API_KEY", "REPLACE_WITH_API_KEY")
```

## Stage, commit, push, PR

```bash
git add <paths>
git commit -F /tmp/msg.txt
GIT_SSH_COMMAND='ssh -o BatchMode=yes' git push -u sky4star <topic-branch>
gh pr create --repo sky4star/<repo> --base main --head <topic-branch> \
  --title "..." --body-file /tmp/pr_body.md
```

Same-repo (fork) PRs are allowed: base `sky4star/<repo>:main`, head the pushed
branch.

## Verify

```bash
gh pr view <n> --repo sky4star/<repo> \
  --json number,state,baseRefName,headRefName,url,changedFiles,additions
```
