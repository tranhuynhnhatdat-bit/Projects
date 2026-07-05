# AGENTS.md

# Python Coding Guidelines

## General Principles

- Write code using simple and readable logic.
- Prefer clarity over cleverness.
- Avoid overly professional or overly abstract code unless specifically requested.
- Keep the existing project structure whenever possible.
- Only modify code that is necessary to complete the requested task.
- Do not refactor unrelated parts of the project.

---

## Code Style

- Keep functions short and focused on one responsibility.
- Use descriptive variable names.
- Avoid deeply nested logic whenever possible.
- Break complicated logic into small helper functions.
- Prefer explicit code over one-line expressions.

Good:

```python
if sharpe_ratio > best_sharpe:
    best_sharpe = sharpe_ratio
    best_model = model
```

Avoid:

```python
best_model = model if sharpe_ratio > best_sharpe else best_model
```

---

## Comments

Always write comments.

Use comments to explain:

- why something is done
- important assumptions
- non-obvious logic
- each major section of the code

Example:

```python
# Scale features using statistics from the training data
scaler = MinMaxScaler()

# Train the model
model.fit(train_inputs, train_targets)

# Evaluate model performance
test_sharpe = compute_filtered_sharpe(...)
```

Avoid commenting every single line.

---

## Existing Code

When editing existing code:

- Preserve function names.
- Preserve inputs and outputs.
- Preserve return values.
- Preserve project structure.
- Do not rename variables unless necessary.
- Do not rewrite working code unnecessarily.

Make the smallest change required.

---

## Error Handling

When fixing bugs:

1. Explain the cause.
2. Fix only the relevant code.
3. Do not rewrite unrelated functions.

---

## Libraries

Do not introduce new libraries unless they provide a significant benefit.

Prefer existing project dependencies.

---

## Explanations

When generating code:

- Explain the reasoning before writing code.
- Keep explanations concise.
- Point out any assumptions.

---

## Teaching Style

Assume the user wants to learn.

Write code that is easy to understand and maintain.

Avoid advanced Python techniques unless they clearly improve readability.

Prefer straightforward loops over complex comprehensions when readability is better.

---

## Output

Unless requested otherwise:

- Produce complete working code.
- Keep formatting clean.
- Keep functions organized.
- Use consistent spacing.
- Use meaningful comments.