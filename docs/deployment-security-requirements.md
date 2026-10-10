# Deployment and security requirements

Status: **requirements and options, written during the follow-up audit.** The service has been run only on localhost for demonstration and
testing; nothing in this document was exposed to a network. Items marked **decision** need the project owner; nothing marked so has been built.

The application today has **no authentication, no authorisation, no TLS and no rate limiting.** That is acceptable for a local demonstration on synthetic data and
is **not** acceptable for any shared or networked use, or for any real customer data.

## 1. What exists now

| Control | State |
|---|---|
| Input validation on `/predict` and batch rows | implemented (finite, bounded numbers; 1–64 character identifiers; timezone handling) |
| Batch size and row limits (2 MB / 500 rows, env-overridable) | implemented |
| Declared request-body limit (batch limit + 100 kB) → 413 | implemented (declared `Content-Length` only) |
| CORS | explicit origin list (default: local Vite ports); methods limited to GET/POST/OPTIONS; headers limited to `Content-Type`, `Accept`; wildcard disables credentials |
| Response headers | `X-Content-Type-Options: nosniff` everywhere; `Cache-Control: no-store` on API responses (not on `/docs`) |
| Interactive docs switch | `API_DOCS=off` removes `/docs`, `/redoc`, `/openapi.json` |
| Trusted PDF reports | the report is built from the stored transaction; the client supplies only an id |
| Model artifact integrity | default model set verified by SHA-256 before the pickled classifier is loaded |
| Secrets in the repository | none tracked (`.env`, `*.db` ignored) |
| Dependency audit | `pip-audit` and `npm audit`: no known vulnerabilities at the audited revision |

## 2. Requirements before any networked deployment

### 2.1 Authentication (**decision**)
Every endpoint except a minimal liveness probe must require an authenticated identity. Options:

| Option | How | Fits when | Trade-offs |
|---|---|---|---|
| **A. Gateway SSO** | a reverse proxy performs OIDC login (for example oauth2-proxy, an API gateway or an identity-aware proxy) and forwards a signed identity; the API accepts requests only from the proxy | the organisation already has an identity provider | least application code; the API must verify the forwarded identity (signed JWT, not a bare header) and must not be reachable except through the proxy |
| **B. Bearer tokens validated by the API** | the dashboard obtains an OIDC access token; FastAPI validates signature, issuer, audience and expiry and reads roles/scopes from it | the dashboard and API are deployed separately or other clients will call the API | needs JWKS handling and token refresh in the SPA; more application code |
| **C. Shared API key** | a static key in a header | service-to-service jobs only | **not suitable for the browser dashboard**: any key shipped to a single-page app is public (`VITE_*` variables are embedded in the bundle). Rotation and per-user audit are poor |

Recommendation: A if an identity provider exists, otherwise B. Do not use C for the dashboard. Passwords must never be stored or handled by this application.

### 2.2 Authorisation and access control (**decision on the role model**)
* Roles at minimum: *analyst* (score, view history and reports), *operator* (batch upload), *administrator* (model information and metrics). Map them from token claims.
* **Object-level control:** customer history, fraud-ring and report endpoints return one customer's behaviour; a caller must be allowed to see *that* customer (portfolio, branch or case assignment). This rule is a business definition and has not been invented here.
* Reports and history must be fetched by an authorised identity and logged (who, what, when). Reports are already generated only from stored data.
* The in-memory customer histories and the `transactions` table hold behavioural data: restrict database and file access to the service account.

### 2.3 Transport security
* TLS 1.2 or newer at the reverse proxy, HSTS on the public host, no plain HTTP; internal hops between proxy, API and database also encrypted where they cross a trust boundary.
* Certificates managed outside the repository.

### 2.4 Rate limiting and abuse resistance (**decision on budgets**)
* Enforce at the gateway, per authenticated identity and per client address; the in-process limiter drafted on the un-merged `keerthan` branch resets on restart and is per process.
* Cost-aware budgets: `/predict` costs about 0.4 s of CPU and grows with a customer's history (6 s at 3,200 transactions); `/predict/batch` costs about 0.4 s per row. Suggested shape: a small per-identity request budget on `/predict`, a much tighter one on `/predict/batch` and `/report/pdf`, a global concurrency cap, `429` with `Retry-After`.
* Request-size limits at the proxy (the application only checks a declared `Content-Length`); upstream timeouts.

### 2.5 CORS
* Exact origins of the deployed dashboard only; never `*` in production. Keep credentials disabled unless cookie sessions are adopted (then add CSRF protection). Prefer serving the dashboard and API from one origin behind the proxy so CORS is not needed.

### 2.6 Secrets
* Database credentials, identity-provider settings and signing keys come from a secret manager or the platform's secret store, not from the repository or the image; `backend/.env` stays ignored; rotate on a schedule and on staff change.
* Frontend `VITE_*` variables are public; never put a credential there.
* Do not print connection strings or request bodies; configuration is not logged today (checked).

### 2.7 Safe production configuration
* `API_DOCS=off`; no `--reload`; `MODEL_SET` left at the verified default; `DATABASE_URL` pointing to PostgreSQL with a least-privilege account (PostgreSQL support is documented as **untested** and must be tested first); schema changes through a migration tool (the current `ALTER TABLE` helper is not one).
* **Single worker process, or externalised state.** Customer histories are held in each process's memory; running several workers makes them diverge. Either run one worker or move the history to a shared store before scaling.
* Model files on a read-only path; pin and verify any artifact set that is allowed to load (`production` set is not pinned today).
* Container or host hardening, non-root user, resource limits, health checks (`/health` already verifies the database).

### 2.8 Logging, monitoring and privacy
* Log at INFO: method, route template, status, latency, authenticated identity, request id. Do **not** log request bodies, full URLs with customer identifiers, or scores tied to a person at INFO. Access logs from the server include customer ids in paths; configure them or the proxy accordingly.
* Audit log for access to history and reports; alerts on repeated 4xx/5xx and on score-distribution drift (each stored transaction already records model set and version).
* Retention: define how long `transactions` rows and reports are kept and how they are erased on request; encrypt backups.
* All data in this repository is synthetic. Before any real data, a data-protection assessment under the applicable law is required; this document is not legal advice.

## 3. Open decisions

1. Which authentication option (2.1) and which identity provider.
2. The role model and the object-level rule for "who may see which customer" (2.2).
3. Rate-limit budgets (2.4) and where they are enforced.
4. Hosting model (single process or shared state) (2.7) and the database to use.
5. Retention and privacy requirements (2.8).
