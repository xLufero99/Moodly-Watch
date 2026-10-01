# Moodly Watch — Agent Notes

AI movie/series/anime recommender. Two independent packages, **no root orchestration**:
`backend/` (Python/FastAPI, uv) and `frontend/` (React 19 + Vite 8, npm). Each has its own
lockfile and commands — always `cd` into the package you are touching.

## Commands

```bash
# backend (cwd MUST be backend/ — see gotcha below)
cd backend && uv sync
uv run uvicorn app.main:app --reload      # http://127.0.0.1:8000, /docs for OpenAPI

# frontend
cd frontend && npm install
npm run dev      # vite dev server
npm run build
npm run lint     # the only configured check in the repo
```

No root `package.json`, `Makefile`, or task runner. No CI, no pre-commit, no `opencode.json` —
nothing enforces lint or tests automatically.

## Backend layout gotcha (will bite you)

- The real app is **`backend/app/`**. `backend/src/moodly_watch_backend/` is a leftover
  `uv_build` scaffold — its `main()` just prints a hello string, and `[project.scripts]
  moodly-watch-backend` points at it. Don't put app code there.
- Imports are **absolute** (`from app.config import settings`), and `app` is **not installed**
  into the venv — `uv_build` only packages `src/`. So backend commands must run with
  `cwd = backend/`. `uv run --project backend ...` from the repo root fails with
  `ModuleNotFoundError: No module named 'app'`.
- `app/config.py:5` uses `env_file=".env"`, which is **relative to cwd** — `.env` is only found
  when you run from `backend/`. There is no `.env` in the repo today (only `.env.example`),
  and every setting has a default, so the app boots fine without it. Add new env vars there
  with a default so imports keep working without a `.env`.
- `app/main.py` enables `allow_origins=["*"]` + `allow_credentials=True` — dev-only, flagged
  as such in a comment. Only `/` and `/health` exist today.
- `app/api/`, `app/core/`, `app/models/`, `app/services/` are empty packages (scaffolded
  layout). `tests/` and `scripts/` are empty dirs; `data/` is gitignored (local data).

## Python tests & linting

pytest and ruff are the project's standards but **are not installed** (absent from `uv.lock`
and `.venv`, even though `.gitignore` lists their caches). Install them **only when explicitly
asked**: `uv add --dev pytest ruff`, then from `backend/`: `uv run pytest` and
`uv run ruff check .`

Never state that tests pass if pytest isn't installed or you didn't actually run them. Don't add
other quality tooling (pre-commit, mypy, etc.) unless asked.

## Frontend ↔ backend contract (the main integration point)

`src/api/client.js:3` has `USE_MOCK = true` — the UI is hardwired to mock data. The backend does
not implement what it calls. Two blockers:

1. **`POST /recommend` does not exist in the backend** (verified 404).
2. **No Vite dev proxy is configured** (`vite.config.js` has only the react + tailwind
   plugins), so `fetch('/recommend')` would hit the Vite dev server on :5173, not FastAPI.

To go live: implement the endpoint, add a `server.proxy` entry pointing at the backend, then flip
`USE_MOCK = false`.

Contract, as declared by the mock layer:
- Request `POST /recommend`, body `{ text, media_types, liked_ids }` → response `{ results: [...] }`.
- Item shape (`src/api/mockData.js` is the authoritative spec):
  `{ id, title, media_type, year, genres[], poster_url (nullable), score 0–1, explanation }`.
- `media_type` ∈ `movie | tv | anime`.
- Typing `error` in the mood input triggers a simulated error while mock mode is on. That's a
  deliberate UI-testing affordance, not a bug — don't "fix" it.

## Frontend conventions

- **Spanish everywhere**: UI strings, code comments, commit messages. Keep it.
- Plain **JSX, no TypeScript** (no tsconfig, eslint only matches `**/*.{js,jsx}`). Adding vitest
  or TS is a decision to raise, not one to make silently.
- Tailwind **v4** is wired via the `@tailwindcss/vite` plugin — there is **no
  `tailwind.config.js`**, and the only CSS rule is `@import "tailwindcss"` in
  `src/index.css`. Styling is inline utility classes in JSX (palette: `zinc` + `violet`).
- App-level UI state is a single `status` string in `App.jsx:25`
  (`'idle' | 'loading' | 'success' | 'error'`), not booleans. `ResultList` switches on it.
- `frontend/README.md` is unmodified Vite template boilerplate — ignore it.

## Workflow

- Work directly on `main` (solo personal project). Only do the phase or block asked in the
  prompt; don't run ahead into later phases.
- Before writing code, give a brief plan and wait for confirmation.
- **Never commit or push** — the user reviews the diff and does that themselves.
- Commit messages: short, Spanish, imperative, with an optional conventional prefix
  (`feat`, `fix`, `chore`, `docs`). Example: `feat: añade endpoint POST /recommend`.
- If the prompt conflicts with a skill or an instruction file, the prompt wins.