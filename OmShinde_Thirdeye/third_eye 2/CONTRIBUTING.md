# Contributing to Third Eye

Thanks for your interest in contributing! This guide outlines how to set up your environment, coding conventions, testing expectations, and documentation practices.

## Setup
- Use Python 3.10+ (tested on 3.13) and create a venv:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- Optional model downloads: `python download_models.py`

## Coding Conventions
- Keep changes minimal and focused; avoid unrelated refactors.
- Use clear function/class docstrings (Google-style preferred).
- Add inline comments only where logic is non-obvious or error-prone.
- Match existing code style; avoid introducing new formatting tools unless configured.

## Tests
- Run `pytest -q`. Test discovery is limited to `tests/`.
- Add tests near the change area. Favor small, precise tests.
- When changing core logic (YOLO, depth, audio, watchdog), add/adjust tests accordingly.

## Documentation
- Update `README.md` and `docs/user-guide.md` for usage changes.
- Add notes in `docs/developer-guide.md` when modifying architecture or threads.
- Keep examples runnable and flags accurate (e.g., `python main.py --headless`).

## Pull Request Checklist
- [ ] Tests pass locally and cover new logic.
- [ ] Docstrings and comments explain tricky areas.
- [ ] Docs updated (README, user guide, developer guide) as needed.
- [ ] No unrelated files or unused code included.

## Communication
- Clearly state assumptions and trade-offs in PR description.
- If changing configuration defaults, call them out explicitly.