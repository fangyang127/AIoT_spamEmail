# Change: add-openspec-ci

## Why
We want to ensure OpenSpec spec deltas and proposals are automatically validated in CI for every pull request so that authors get fast feedback and proposals follow the required format (scenarios, delta headers, etc.). This reduces manual review cycles and prevents invalid proposals from being merged.

## What Changes
- Add a GitHub Actions workflow (or equivalent CI) that runs `openspec validate <change-id> --strict` on PRs that touch `openspec/changes/` or `openspec/specs/`.
- Document the CI requirement in `openspec/project.md` and PR templates.
- Add a sample `changes/add-openspec-ci/specs/ci/spec.md` delta describing the validation requirement.

## Impact
- Affected specs: CI/ops (new capability)
- Affected code: CI configuration (e.g., `.github/workflows/openspec-validate.yml`), PR templates
- Non-breaking: this adds validation and CI checks but does not change runtime behavior of deployed systems.
