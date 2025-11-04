## ADDED Requirements

### Requirement: OpenSpec CI Validation
The project SHALL run OpenSpec validation for any proposed changes under `openspec/changes/` or any changes to `openspec/specs/` as part of CI.

#### Scenario: Pull request triggers validation
- **WHEN** a PR modifies files under `openspec/changes/` or `openspec/specs/`
- **THEN** CI runs `openspec validate <change-id> --strict` for each matching change directory
- **THEN** the CI status is reported back to the PR

#### Scenario: Validation failure blocks merge
- **WHEN** `openspec validate` fails for a change
- **THEN** CI fails and the PR merge is blocked until the change is corrected and validation passes
