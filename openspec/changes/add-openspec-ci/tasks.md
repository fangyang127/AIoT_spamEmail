## 1. Implementation
- [ ] 1.1 Create CI workflow file to run `openspec validate <change-id> --strict` on PRs
- [ ] 1.2 Update `openspec/project.md` to document CI requirement (done)
- [ ] 1.3 Add PR template note to remind authors to include change-id when proposing spec changes
- [ ] 1.4 Add example delta under `specs/ci/spec.md` (done)
- [ ] 1.5 Run `openspec validate add-openspec-ci --strict` and iterate until validation passes

## 2. Optional
- [ ] 2.1 Add a GitHub Action status check to block merges when validation fails
- [ ] 2.2 Add automation to auto-run `openspec validate` on push to feature branches
