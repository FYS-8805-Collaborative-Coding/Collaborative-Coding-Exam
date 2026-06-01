# Contributing

Thank you for contributing! This guide explains how we collaborate on this project.

---

## How we work

All changes go through **branches and pull requests** — no one pushes directly to `main`. This keeps the history clean and ensures every change gets reviewed.

The general flow is:

1. Open an issue describing what you want to change
2. Create a branch for your work
3. Make commits on that branch
4. Open a pull request towards `main`
5. Get a review, address any feedback
6. Merge once approved

---

## Step by step

### 1. Open an issue

Before starting work, open an issue on GitHub describing the change. Note the issue number (e.g. `#5`) — you'll use it in your commit message.

### 2. Create a branch

Create a new branch for your change. Use a short, descriptive name:

```bash
git checkout -b your-branch-name
```

Each pull request should come from its own branch — this way you can have multiple open pull requests without them interfering with each other.

### 3. Commit your changes

Write clear commit messages. If your commit addresses an issue, reference it so GitHub closes it automatically on merge:

```
add preprocessing step for missing values; fixes #5
```

Keywords GitHub recognizes: `fixes`, `closes`, `resolves` (case insensitive).

### 4. Push and open a pull request

Push your branch to GitHub:

```bash
git push origin your-branch-name
```

Then open a pull request towards `main` on GitHub. Give it a descriptive title and a short description of what it does and why.

If your work is not finished yet, open a **draft pull request** — this signals that it is not ready to merge but lets others follow along and give early feedback.

### 5. Code review

At least one other group member should review each pull request before it is merged.

**As a reviewer:**
- Be kind and constructive — the goal is collaborative learning, not gatekeeping
- Check that the title and description are clear
- Look through the changes and leave comments where something is unclear or could be improved
- For small issues like typos, you can suggest a direct fix using the "±" button in the "Files changed" tab — the author can accept it with one click
- Use "Request changes" if something needs to be addressed before merging

**As the author:**
- Respond to comments and make improvements by adding new commits to the same branch — the pull request updates automatically, so do not close and reopen it
- Once all comments are addressed, let the reviewer know

### 6. Merge

Once the pull request is approved, merge it into `main`. The branch can then be deleted.

---

## Protected branch

`main` is a protected branch — it cannot be pushed to directly. All changes must go through a pull request. This is set up under Settings → Branches.

---

## External contributors

If you do not have write access to this repository, you can still contribute via a fork:

1. Fork the repository to your own GitHub account
2. Clone your fork and create a branch
3. Make your changes and push to your fork
4. Open a pull request from your fork towards `main` in this repository

The same review process applies — your PR will be reviewed before merging.

---

## Questions?

Open an issue or leave a comment on the relevant pull request.