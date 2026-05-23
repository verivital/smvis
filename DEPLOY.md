# Deploying smvis to Google Cloud Run

This deploys the smvis web app (Dash) as a container on **Google Cloud Run**,
with the **Linux nuXmv** binary bundled so spec-checking and the interactive
terminal work server-side. Cloud Run scales to zero, so for a class of ~20–30
intermittent users the cost is effectively **$0** (well within the always-free
tier).

## What gets deployed

- The Dash app served by **gunicorn** (`smvis.wsgi:server`).
- The **Linux x86_64 nuXmv 2.1.0** binary, extracted from
  `bin/nuxmv/nuXmv-2.1.0-linux64.tar.xz` during the image build and placed at
  `/app/bin/nuxmv/nuXmv` (the app finds it via `SMVIS_NUXMV_PATH`).
- The example models in `examples/`.

### Single-instance by design

Two pieces of server state are **module-level globals** in `app.py`: the nuXmv
interactive terminal session (`_nuxmv_session`) and the SHA-256 result cache
(`_compute_cache`). For these to stay coherent, the app runs as **one gunicorn
worker** and we deploy with **`--max-instances 1`**. Concurrency for multiple
users comes from gunicorn threads, which is plenty for this workload. (Batch
spec-checking spawns a fresh nuXmv subprocess per request and is stateless — only
the interactive terminal is truly single-session.)

---

## Prerequisites (one-time)

1. **Install the gcloud CLI**: <https://cloud.google.com/sdk/docs/install>
2. **Create / pick a project** and enable billing (required even to use the free
   tier — you won't be meaningfully charged at this scale):
   ```powershell
   gcloud auth login
   gcloud projects create smvis-eecs6315           # or reuse an existing project
   gcloud config set project smvis-eecs6315
   ```
   Then link a billing account in the Cloud Console (Billing → link project).
3. **Enable the required APIs**:
   ```powershell
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
   ```

---

## Deploy

From the `smvis/` directory (the one containing the `Dockerfile`):

```powershell
gcloud run deploy smvis `
  --source . `
  --region us-central1 `
  --allow-unauthenticated `
  --memory 1Gi `
  --cpu 1 `
  --max-instances 1 `
  --concurrency 80 `
  --timeout 300
```

What the flags do:

| Flag | Why |
|------|-----|
| `--source .` | Cloud Build builds the `Dockerfile` for you — no local Docker push needed |
| `--allow-unauthenticated` | Public URL so students can open it without a Google login |
| `--max-instances 1` | Keeps the single nuXmv session + cache in one instance (see above) |
| `--memory 1Gi` | Headroom for z3 + dd + a nuXmv subprocess |
| `--concurrency 80` | Many simultaneous users share the one instance |
| `--timeout 300` | Allows longer model-checking requests |

The first build takes a few minutes. When it finishes, gcloud prints the
**Service URL** (e.g. `https://smvis-xxxxx-uc.a.run.app`). Share that link with
the class.

### Redeploying after changes

Re-run the exact same `gcloud run deploy` command. Each run builds a new revision
and shifts traffic to it.

---

## Cost & behavior

- **Scale to zero**: with `--min-instances 0` (the default), you pay nothing when
  no one is using it. The first request after idle incurs a cold start of a few
  seconds.
- **Free tier**: Cloud Run includes 2M requests, 180k vCPU-seconds, and 360k
  GiB-seconds per month — a class session stays well inside it.
- **Want zero cold start during class?** Add `--min-instances 1` (keeps one
  instance warm — small cost while warm) and, if the interactive terminal feels
  laggy between commands, `--no-cpu-throttling` (CPU stays allocated between
  requests). Drop both back after class to return to ~$0.

---

## Local smoke test (optional, requires Docker)

Build and run the exact production image locally before deploying:

```powershell
docker build -t smvis .
docker run --rm -p 8080:8080 smvis
# open http://localhost:8080
```

Confirm nuXmv is wired up correctly:

```powershell
# Should print the dynamic libraries it needs (all must resolve — no "not found")
docker run --rm --entrypoint ldd smvis /app/bin/nuxmv/nuXmv

# Should print the nuXmv banner / version
docker run --rm --entrypoint /app/bin/nuxmv/nuXmv smvis -h
```

If `ldd` reports a missing library, add the matching Debian package to the
`apt-get install` list in the `Dockerfile` (runtime stage) and rebuild.

---

## Optional: continuous deploy from GitHub

If you push `smvis/` to GitHub and want auto-deploy on every push:

```powershell
gcloud builds triggers create github `
  --name smvis-deploy `
  --repo-name <your-repo> --repo-owner <your-user> `
  --branch-pattern "^main$" `
  --build-config cloudbuild.yaml
```

…with a `cloudbuild.yaml` that builds the image and runs `gcloud run deploy
--image`.

> ⚠️ **Do not commit the nuXmv binary/tarball to this repo.** `bin/` is
> gitignored on purpose: the GitHub remote is **public** and nuXmv's license
> forbids redistribution. A GitHub-triggered build therefore won't have the
> tarball. Options for continuous deploy: (a) keep using local `--source`
> deploys from a checkout that has the tarball (simplest); (b) store the tarball
> in a private GCS bucket and `gsutil cp` it during the Cloud Build step; or
> (c) host nuXmv in a private repo/Artifact Registry. The `--source` deploy
> above needs no GitHub and is the recommended path.

---

## Troubleshooting

- **nuXmv "not found" in the app** → check the build logs show the binary copied
  to `/app/bin/nuxmv/nuXmv`, and that `SMVIS_NUXMV_PATH` is set (it is, in the
  Dockerfile). Verify with the `ldd` command above.
- **Interactive terminal resets** → it's a single shared session by design; only
  one user should drive it at a time, and scale-to-zero will end an idle session.
  Use `--min-instances 1` for a smoother live demo.
- **Out of memory** → bump `--memory 2Gi`.
- **Architecture note** → nuXmv ships x86_64 only; the image is pinned to
  `linux/amd64`. Do not deploy to ARM hosts.
