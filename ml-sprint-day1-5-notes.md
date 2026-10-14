# 20-Day ML Sprint — Notes (Days 1, 4–5)

## DAY 1 — Git Fundamentals

### The core loop
Every day of this sprint, every project you'll ever build, uses this exact sequence:

```
edit file → git add <file> → git commit -m "message" → git push
```

- **`git add`** does NOT save anything. It *stages* a file — marks it as "include this in my next commit." Nothing is permanent yet.
- **`git commit`** takes whatever is staged and permanently saves it as a snapshot in project history, with a message.
- **`git push`** sends your local commits to GitHub (or wherever `origin` points).

### File states (the thing that actually matters)
A file moves through these states:

```
untracked → staged → committed → modified (if edited again) → staged → committed ...
```

Check which state you're in with `git status` — **read the first line first** (what branch you're on), then the file list.

### Branches and merging
- `git checkout -b <name>` creates AND switches to a new branch
- `git checkout <name>` switches to an existing branch (no `-b`)
- Two branches that each change the *same line* of a file differently → **merge conflict** when you try to combine them
- `git merge <branch>` — if there's a conflict, Git marks the file:
  ```
  <<<<<<< HEAD
  (what's on your current branch)
  =======
  (what's coming in from the other branch)
  >>>>>>> other-branch
  ```
  Edit the file by hand, delete the `<<<<<<<` / `=======` / `>>>>>>>` markers, keep what you want, then `git add` + `git commit` to finish the merge.
- **`HEAD`** = Git's pointer to wherever you currently are (your current branch/commit). It's relative to you, not fixed.
- `git log --oneline --graph --all` — draws your branch history as an ASCII graph. Good for seeing the actual shape of a merge.
- `git merge --abort` — escape hatch if a conflict is too messy, resets to before the merge attempt.

### Mistakes made (and the actual lesson in each)
| Mistake | Root cause | Lesson |
|---|---|---|
| `git add <file>` typed literally | `<file>` in docs means "put your real filename here" — never type the brackets | Angle brackets = placeholder convention, everywhere in tech docs |
| `git -commit -m "..."` failed | Dash in wrong place — options go *after* the command | Read error messages bottom-to-top for the real cause |
| `git checkout -b feature -b` | Typed a space instead of a hyphen (`feature-b` ≠ `feature -b`) | Precision matters — Git parses exactly what you type |
| Edited file while on `main` instead of a feature branch | Didn't check `git status`'s first line before editing | **Always check which branch you're on before editing** |

---

## DAY 4–5 — Logistic Regression From Scratch (NumPy, no sklearn)

### Why sigmoid, not raw `z`?
Raw `z = w·x + b` is unbounded (−∞ to +∞) — there's no natural scale to measure "how wrong" a prediction is. Sigmoid squashes `z` into (0, 1), turning it into a genuine **probability**. That probability is what lets you use a real loss function (binary cross-entropy) with proper mathematical grounding — and that's what makes the *gradients* meaningful for learning.

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

- `sigmoid(0) = 0.5` → net evidence is zero, the model has no opinion
- `sigmoid(+∞) → 1`, `sigmoid(-∞) → 0` → saturates as evidence grows

### Why `w·x + b` and not `w + x + b`?
Multiplication gives each feature its own independent "dial" (weight) the model can tune. Addition would force every feature to matter equally (coefficient of 1), and the model couldn't learn that one feature matters more than another.

### The pipeline, step by step

```python
# 1. Forward pass
def predict_proba(X, w, b):
    z = X @ w + b
    return sigmoid(z)

# 2. Loss — binary cross-entropy
def binary_cross_entropy(y, p):
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)  # avoid log(0) = -infinity -> NaN crash
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

# 3. Gradients (derived from calculus, but collapses to this clean form)
def compute_gradients(X, y, p):
    n = X.shape[0]
    error = p - y                  # how wrong each prediction is, with sign
    dw = (1/n) * X.T @ error       # average "wrongness alignment" per feature
    db = (1/n) * np.sum(error)
    return dw, db

# 4. Training loop
w = np.zeros(n_features)
b = 0.0
learning_rate = 0.01
losses = []

for i in range(epochs):
    p = predict_proba(X, w, b)
    loss = binary_cross_entropy(y, p)
    losses.append(loss)
    dw, db = compute_gradients(X, y, p)
    w = w - learning_rate * dw
    b = b - learning_rate * db
```

**Mental model for the gradient:** `error = p - y` tells you how wrong each prediction is, with direction. `X.T @ error` averages, across all examples, how much each feature's value "lines up" with that error — that's what tells each weight which way to move. It's a vectorized shortcut for something you could write as a `for` loop over features.

### Landmark number: `-log(0.5) ≈ 0.693`
This is the loss of a model that knows nothing (predicts 0.5 for everything). **Any trained model with a loss higher than this is worse than random guessing** — a permanent sanity check.

### The bug that mattered most: unscaled features
**Symptom:** loss curve drops steeply for ~100 epochs, then goes almost flat ("cliff then crawl").

**Cause:** Features on very different scales (e.g., BMI ~15–40 vs. HbA1c ~4–13) get hit with the *same* learning rate. Gradient descent races down the direction of the large-scale feature and crawls on the small-scale one.

**Fix — standardize before training:**
```python
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_scaled = (X - X_mean) / X_std
```

**Result on real data:** final loss went from **0.437** (unscaled, stalled) → **0.21** (scaled, 3000 epochs, still converging smoothly).

**Bonus lesson:** weight *signs/magnitudes are not trustworthy for comparison* until features are scaled. Unscaled run gave BMI a negative weight (artifact of mid-correction gradient descent); scaled run gave BMI a small positive weight and showed HbA1c pulling ~4–5x harder — the real, interpretable relationship.

### Rule to keep forever
**Loss going down ≠ the model is good.** Loss tells you the model fits better than before — it says nothing about accuracy, precision/recall, or whether the model is just exploiting class imbalance (this dataset: 85% not-High-risk / 15% High-risk). That check is next: turn probabilities into class predictions, build a confusion matrix, and find out if 0.21 loss is actually *useful*.

---

## Debugging habits that actually got built this week
1. **Read the first line of `git status` before anything else** — branch name, then file state.
2. **Read tracebacks bottom-up** — the last line is almost always the real error, in your own file.
3. **`.unique()` before filtering a text column** — case sensitivity (`'High'` vs `'high'`) causes silent empty-result bugs, not crashes.
4. **Match plot type to the question** — `plt.plot()` connects points in row order (use for ordered/time data); `plt.scatter()` shows relationship between two variables regardless of row order.
5. **Form a hypothesis before running the fix** — predict what should happen, then check if the data agrees. (This caught the "features vs. scaling" question properly instead of guessing.)
