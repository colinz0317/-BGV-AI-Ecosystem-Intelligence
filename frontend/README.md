# BGV Platform Frontend

React (Vite) UI for the BGV AI Ecosystem Intelligence platform. Consumes the FastAPI backend in `../backend`.

For end-to-end setup (backend + data files + env vars), see the root [README.md](../README.md).

## Run Locally

Backend must be running on `http://127.0.0.1:8000` first (see root README).

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
npm run build      # production bundle → dist/
```

## Structure

- `bgv_platform_prototype.jsx` — single-file app (all tabs, modal, state)
- `src/main.jsx` — Vite entry, mounts `<BGVPlatform />`
- `index.html` — HTML shell
- `vite.config.js` — Vite config

## Backend API Endpoints Consumed

All calls go to `http://127.0.0.1:8000` (hardcoded; CORS `*` enabled on the backend).

| Endpoint | Used By |
|---|---|
| `GET /api/companies` | Opportunity Matrix — row/column headers |
| `GET /api/matrix` | Opportunity Matrix — heatmap scores |
| `GET /api/evidence?company=&theme=` | Matrix cell drill-down (events + sub-themes) |
| `GET /api/gaps` | Strategic Gaps Library |
| `GET /api/portfolio-companies` | Portfolio Matcher dropdown |
| `GET /api/portfolio/{company}` | Portfolio Matcher — company description |
| `GET /api/match?company=&top_k=` | Portfolio Matcher — GMS results |
| `GET /api/recommendation?company=` | Generate Report — LLM-generated investment memo |
| `GET /api/settings`, `PUT /api/settings` | Settings tab |

## Tabs

- **Opportunity Matrix** — ISS heatmap (hyperscalers × themes), data source priority selectors (High/Mid/Low), click a cell for evidence
- **Strategic Gaps Library** — filterable list of extracted capability gaps
- **Portfolio Matcher** — select a BGV portfolio company, Top N Gaps slider, run GMS, generate PDF report with LLM memo
- **Settings** — ISS weight display (read-only configuration reference)
