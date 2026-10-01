# Development instructions

This release contains the paper's Exp1 and final five-task Exp2. Real-robot and
Agent+VLA implementations are outside this distribution.

- Use `pixi run --locked` with the appropriate checked-in environment.
- Preserve scientific policy source, fixed cases, prompts, and artifact hashes.
- Keep framework implementation distinct from construction-agent policy designs.
- Never start paid API calls, training, or formal evaluations as an installation
  check. Use offline tests and released result aggregation.
- Keep credentials in environment variables and filesystem roots configurable.
- Run the release audit, asset verification, and relevant tests after changes.
- Record evidence and limitations; do not claim an unexecuted experiment passed.
