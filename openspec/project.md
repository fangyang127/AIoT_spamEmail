# Project Context

## Purpose
This repository contains coursework for the AIoT (Artificial Intelligence + IoT) homework 3 assignment. The goal of this project is to exercise building, testing and documenting IoT-related components and automation scripts while following a spec-driven development approach using OpenSpec.

## Tech Stack
- Primary runtime: Node.js (recommended >=16.x)
- Package manager: npm
- CLI tooling: `@fission-ai/openspec` (OpenSpec CLI) for spec-driven proposals and validation
- Optional: TypeScript for type safety, Jest for unit tests, ESLint + Prettier for linting/formatting
- Target devices / protocols (domain): microcontrollers/embedded devices (e.g., ESP32/STM32), MQTT for messaging, and optionally Raspberry Pi for gateways. (Assumption: this is an AIoT course repo — confirm if different.)

## Project Conventions

### Code Style
- Use Prettier for formatting (`prettier --write`) and ESLint for linting if code is JavaScript/TypeScript.
- Filename casing: kebab-case for scripts and specs, PascalCase for React components (if any), camelCase for variables/functions.
- Commit messages: follow Conventional Commits (e.g., `feat: add sensor reader`, `fix: correct mqtt topic`).

### Architecture Patterns
- Keep modules single-purpose and minimal. Each capability should map to a folder under `specs/`.
- Prefer simple, synchronous flows for device onboarding. Use message brokers (MQTT) for decoupling sensors and backends.

### Testing Strategy
- Unit tests: use Jest for JS/TS modules. Keep tests near code (e.g., `__tests__/` or `.test.ts` files).
- Integration testing: prefer small CI jobs that spin up local services (e.g., MQTT broker via Docker) for integration checks.
- Spec validation: use `openspec validate <change-id> --strict` before requesting approvals.

### Git Workflow
- Branching: `main` (release/stable), `dev` (integration), feature branches: `feat/<short-desc>`.
- Pull requests: open PRs against `dev` or `main` depending on your workflow. Include the related `openspec` change-id in PR description when applicable.
- Reviews: require at least one review before merging to `dev` and two for `main` (course/team policy suggestion).

## Domain Context
- The project focuses on combining AI techniques with IoT data collection and device communication. Typical concerns:
	- Data ingestion latency and reliability (MQTT QoS)
	- Device provisioning and secure keys
	- Lightweight ML inference on edge devices vs. cloud inference

## Important Constraints
- Development is performed on Windows (PowerShell) and tested in cross-platform CI.
- Keep external dependencies minimal; prefer lightweight libraries for embedded compatibility.

## External Dependencies
- MQTT broker(s) (e.g., Mosquitto)
- Cloud endpoints (if any) — document connection strings/secrets in a secure vault, not in repo.
- `@fission-ai/openspec` CLI for spec management and validation

## Development Commands (PowerShell)
```powershell
# install dependencies
npm install

# run spec validation for a change
openspec validate <change-id> --strict
# if openspec isn't on PATH, run with npx:
npx @fission-ai/openspec validate <change-id> --strict

# initialize openspec (already done once):
openspec init
```

## Assumptions
- This is an AIoT homework repository (named `hw3` in the workspace path). If the repo is for production or a different assignment, tell me and I will update details (stack, device targets, CI config).

## Where to add project knowledge
- Add capability-specific context under `openspec/specs/<capability>/spec.md`.
- Add proposed changes under `openspec/changes/<change-id>/` following OpenSpec conventions.

## Spam Email Classifier (Project-specific OpenSpec `project.md`)

This section describes the `Spam Email Classifier` project which is part of the IoT Data Analysis and Applications course. It focuses on building a reproducible machine learning pipeline (preprocessing, training, evaluation) using Python and scikit-learn, and exposing results and demos via Streamlit.

Overview
--------
- Name: Spam Email Classifier
- Course: IoT Data Analysis and Applications
- Short description: A reproducible pipeline to preprocess email text, train a scikit-learn classifier to detect spam vs. ham, evaluate models, version model artifacts, and provide an interactive Streamlit app for exploration and demo.

Goals
-----
- Build a clear, testable preprocessing pipeline for email text (tokenization, normalization, feature extraction e.g., TF-IDF).
- Train one or more scikit-learn models (e.g., Logistic Regression, Random Forest, SVM) with cross-validation and hyperparameter tuning.
- Persist model artifacts and metadata (model version, training data hash, metrics) for reproducibility.
- Provide an interactive Streamlit visualization for model predictions, dataset exploration, and performance metrics.
- Integrate OpenSpec workflow for changes: proposals, spec deltas, validation and CI checks.

Tech Stack
----------
- Language: Python 3.10+ (recommendation; adjust as needed)
- Libraries: scikit-learn, pandas, numpy, joblib (or dill) for model serialization, scikit-learn-pipeline, nltk/spacy (optional) for NLP preprocessing, streamlit for UI
- Testing: pytest
- Linting/formatting: flake8 or pylint, black
- Packaging / environment: pip + requirements.txt, optional Poetry
- CI: GitHub Actions (recommended) or other CI to run tests and OpenSpec validation

Directory Structure
-------------------
The repository should be organized for clarity and OpenSpec collaboration:

```
.
├── data/                       # Raw and processed sample data (ignored in .gitignore for large files)
├── src/                        # Source code
│   ├── preprocessing/          # Tokenizers, cleaners, feature extraction (TF-IDF) pipelines
│   ├── models/                 # Model training code, hyperparameter tuning scripts
│   ├── eval/                   # Evaluation and metrics utilities
│   └── app/                    # Streamlit demo app
├── experiments/                # Notebooks or experiment logs
├── models/                     # Serialized model artifacts (.joblib) and metadata (model card)
├── tests/                      # pytest tests
├── requirements.txt            # pinned deps for reproducibility
├── setup.py / pyproject.toml   # optional packaging
├── Makefile                    # helper commands (optional)
└── openspec/
	 ├── project.md              # this file
	 ├── specs/                  # current truth - deployed/expected capabilities
	 │   └── [capability]/
	 │       └── spec.md
	 └── changes/                # proposals under review
		  └── [change-id]/        # proposal.md, tasks.md, specs/ delta files
```

Conventions
-----------
- Python modules: snake_case filenames and snake_case functions.
- Class names: PascalCase.
- Spec and change naming: kebab-case; change-id must be verb-led (e.g., `add-model-logging`, `update-preprocessing-tokenizer`).
- Model artifacts: store with semantic filename `model--<model-name>--v<semver>.joblib` and include a `metadata.json` (schema: model_name, version, training_date, metrics, data_hash).
- Data privacy: do not commit PII or real email datasets to repo. Use sample or synthetic data for demos and put real data behind protected storage.
- Tests: place unit tests under `tests/` with `test_*.py` naming. CI must run `pytest`.

Collaboration Workflow (OpenSpec + Git)
-------------------------------------
1. Propose changes
	- For any new capability or breaking change, create a change folder: `openspec/changes/<change-id>/`.
	- Include `proposal.md` (why/what/impact), `tasks.md` (implementation checklist), and `specs/<capability>/spec.md` delta files.
	- Follow OpenSpec requirement formatting: use `## ADDED|MODIFIED|REMOVED Requirements` and ensure each `### Requirement:` has at least one `#### Scenario:`.

2. Validate spec
	- Run `openspec validate <change-id> --strict` locally or in CI before requesting approval.
	- Example (PowerShell):
	 ```powershell
	 npx @fission-ai/openspec validate add-model-logging --strict
	 ```

3. Implement
	- After proposal approval, implement tasks listed in `tasks.md`. Keep commits small and reference the `change-id` in PR description.
	- Add unit tests for preprocessing, model training (small smoke tests), and app routes/components.

4. CI and checks
	- CI should run:
	  - `pip install -r requirements.txt` (or use cached environment)
	  - `pytest`
	  - `npx @fission-ai/openspec validate <change-id> --strict` for PRs affecting `openspec/` files
	  - Optional: static checks (black --check, flake8)

5. Merge and archive
	- After merge and deployment, archive change: move `changes/<id>/` to `changes/archive/YYYY-MM-DD-<id>/` and update `specs/` as needed.

Developer Commands (PowerShell)
-------------------------------
```powershell
# create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install deps
pip install -r requirements.txt

# run tests
pytest -q

# run Streamlit demo
streamlit run src/app/main.py

# validate an OpenSpec change
npx @fission-ai/openspec validate <change-id> --strict
```

References
----------
- OpenSpec guide: `openspec/AGENTS.md` and the repository `openspec/` folder (use `openspec validate` and `openspec list` as needed)
- scikit-learn documentation: https://scikit-learn.org/stable/
- Streamlit docs: https://docs.streamlit.io/
- Best practices for ML reproducibility: model versioning, deterministic pipelines, record random seeds, store training data hashes and environment (requirements.txt / lockfile).

Notes and Assumptions
---------------------
- This `project.md` assumes Python-based workflow. If you prefer a Node.js or mixed-stack approach for the demo, tell me and I will adapt.
- Real email datasets may contain private data; treat them according to course / institutional rules.

