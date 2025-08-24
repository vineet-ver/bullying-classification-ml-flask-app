# Bullying Classification ML Flask App

Simple Flask app that predicts whether text is bullying (1) or non-bullying (0) using a pickled sklearn model and TF-IDF vectorizer.

## CI/CD: GHCR + Cloud Run

This repository includes a GitHub Actions workflow (`.github/workflows/ghcr-cloud-run.yml`) which:

- Builds a Docker image and pushes it to GitHub Container Registry (GHCR).
- Deploys that image to Google Cloud Run.

Required GitHub repository secrets (Repository settings → Secrets → Actions):

- `GCP_SA_KEY` — JSON key for a Google Cloud service account with the following roles: `roles/run.admin`, `roles/iam.serviceAccountUser`, and `roles/storage.admin` (or narrower permissions if you prefer).
- `GCP_PROJECT` — Your GCP project ID.
- `GCP_REGION` — Cloud Run region (example: `us-central1`).
- `CLOUD_RUN_SERVICE` — The Cloud Run service name to deploy (example: `flask-bullying-app`).

How to create the service account key:

1. In the Google Cloud Console, create a Service Account under IAM & Admin → Service Accounts.
2. Grant the service account the roles: `Cloud Run Admin`, `Service Account User`.
3. Create and download a JSON key, then add its contents as the `GCP_SA_KEY` secret in GitHub.

When you push to `main`, the workflow will build, push, and deploy automatically.
