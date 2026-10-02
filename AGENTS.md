# Moodly Watch — Agent Notes

AI movie/series/anime recommender. Two independent packages, **no root orchestration**:
`backend/` (Python/FastAPI, uv) and `frontend/` (React 19 + Vite 8, npm). Each has its own
lockfile and commands — always `cd` into the package you are touching.

For where the project stands — phases, measured numbers, what's left — see
`docs/ESTADO.md`. This file is about conventions and traps, not about progress.

## Commands

```bash
# backend (cwd MUST be backend/ — see gotcha below)
cd backend && uv sync
uv run uvicorn app.main:app --reload      # http://127.0.0.1:8000, /docs for OpenAPI
uv run python -m scripts.build_catalog   # raw JSON -> data/processed/catalog.parquet
uv run python -m scripts.build_index      # catalog.parquet -> data/index/ (ChromaDB, ~36 min)
uv run python -m scripts.search_demo "una frase de estado de animo"

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
  as such in a comment. Routes today: `/`, `/health` and `POST /recommend`.
- `app/api/`, `app/models/`, `app/services/` and `scripts/` are all populated. Beware: FastAPI
  0.142 nests routers as an `_IncludedRouter` object, so iterating `app.routes` no longer lists
  a router's endpoints. Use `app.openapi()["paths"]` to see what is actually served.
- `data/index/` (ChromaDB) and `data/processed/` are gitignored. `data/index/` must be
  rebuilt, not committed.

## Python tests & linting

pytest and ruff are the project's standards and are **installed** (dev group). From
`backend/`: `uv run pytest` and `uv run ruff check .`

Never state that tests pass if you didn't actually run them. Don't add other quality tooling
(pre-commit, mypy, etc.) unless asked.

`ruff format` is **not** the standard here: the existing files don't pass it either. Use
`ruff check .` and leave formatting alone.

## Embeddings: torch must stay CPU-only

`torch` is pinned to the PyTorch CPU index, because on Linux PyPI serves CUDA wheels that
are several GB and are useless here (`torch+cpu` is 737 MB). `pyproject.toml` carries:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

`explicit = true` is load-bearing. `uv add ... --index pytorch-cpu=<url>` alone does **not**
set it, and the resolution then fails because the PyTorch index becomes the only registry
and can't find the other packages. Don't remove the block, and don't re-add torch by hand.

Two things about ChromaDB metadata that were verified by running it, not by reading docs:
it **rejects `None`** (the Rust layer raises even though the Python validator allows it),
and it **rejects `np.int64`**. Cast to native Python types and drop null keys — no sentinel
values.

## Frontend ↔ backend contract (the main integration point)

`src/api/client.js:3` has `USE_MOCK = false`, and `vite.config.js` already proxies
`/recommend` and `/health` to `http://localhost:8000`. The wiring is done and the app runs
end to end. What is **not** done is the backend logic: `POST /recommend` exists but is served
by `app/services/mock_recommender.py`, so it returns mock data.

To go live for real: query the ChromaDB index built by `scripts/build_index.py`, then use
Groq for the `explanation` field, which nothing generates yet (`groq_api_key` is unused).

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