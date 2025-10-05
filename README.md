# Kalman 1D Velocity Motion


Install necessary libralies.

```terminal
uv pip sync
uv run pre-commit install
```

If you have not installed uv, you can use this:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

You can lanch the application.

```bash
uv run streamlit run app.py 
```

I am using WSL2, Ubuntu 24.02 and following messages are shown in terminal.

> You can now view your Streamlit app in your browser.
>
>  Local URL: http://localhost:8501
>  Network URL: http://172.29.73.34:8501

## Usage

```
PYTHONPATH=. uv run tools/sample_generator.py --help
```

## Memo to Deploy to Google Cloud Platform

- Set your project. Create project if necessary.

```bash
gcloud auth login
gcloud projects create [YOUR_PROJECT_ID] --name="[PROJECT_NAME]"
gcloud config set project [YOUR_PROJECT_ID]
gcloud auth application-default set-quota-project [YOUR_PROJECT_ID]
gcloud config set run/region asia-northeast1
```

- Check billing account and link the project

```shell
gcloud beta billing accounts list
gcloud beta billing projects link [YOUR_PROJECT_ID] --billing-account [BILLING_ACCOUNT_ID]
```

- Create Artifact Registry

```shell
gcloud artifacts repositories create my-repo --repository-format=docker --location=asia-northeast1
```

- Give Authentification to current account (e-mail accress) if necessary.

```shell
gcloud projects add-iam-policy-binding [YOU_PROJECT_ID] \
  --member="user:[CURRENT_ACCOUNT]" \
  --role="roles/editor"
```

- Build docker image and push.

```shell
gcloud builds submit --tag asia-northeast1-docker.pkg.dev/[YOUR_PROJECT_ID]/my-repo/streamlit-app
```


- Deploy

```shell
gcloud run deploy streamlit-app \
  --image asia-northeast1-docker.pkg.dev/[YOUR_PROJECT_ID]/my-repo/streamlit-app \
  --platform managed \
  --port 8501 \
  --allow-unauthenticated
```

- Build and deployment process are written in cloudbuild script.



### Cloudbuild

- https://cloud.google.com/build/docs/securing-builds/configure-user-specified-service-accounts?hl=ja

- Create a service account for cloudbuild.

```shell
gcloud iam service-accounts create cloud-build-sa --display-name="Cloud Build Service Account"
```

- Add roles to the service account. 

```
gcloud projects add-iam-policy-binding [YOUR_PROJECT_ID] \
  --member="serviceAccount:cloud-build-sa@[YOUR_PROJECT_ID].iam.gserviceaccount.com" \
  --role="roles/cloudbuild.builds.builder"
```

- Also add roles of Cloud Run (deploy) and Artifact Registry (read/write).

```shell
gcloud projects add-iam-policy-binding [YOUR_PROJECT_ID] \
  --member="serviceAccount:cloud-build-sa@[YOUR_PROJECT_ID].iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"
gcloud projects add-iam-policy-binding [YOUR_PROJECT_ID] \
  --member="serviceAccount:cloud-build-sa@[YOUR_PROJECT_ID].iam.gserviceaccount.com" \
  --role="roles/run.admin"
```

```shell
gcloud iam service-accounts add-iam-policy-binding [target_service_account_email] \
  --member="serviceAccount:[service_account_email_granting_to_target]" \
  --role="roles/iam.serviceAccountUser" \
  --project=[YOUR_PROJECT_ID]
```

```shell
gcloud builds submit --config cloudbuild.yaml --service-account=projects/[PROJECT_ID]/serviceAccounts/[SERVICE_ACCOUNT_EMAIL]
```

## Stop /close

You have a couple of options for disabling your Cloud Run services, which effectively stops any running instances. The primary method is to set the scaling of your service to zero instances. This allows existing requests to complete but prevents new ones from being processed.


```shell
gcloud beta run services update SERVICE_NAME --scaling=0 --region=REGION_NAME --project=PROJECT_ID
```


## Development MEMO

### Unit test

Add pytest in independencies.

```shell
uv add pytest --dev
uv add pytest-cov --dev
```

Run unit tests.

```shell
# with coverage and detail output
uv run pytest --cov=src -v

# specific file
uv run pytest tests/test_kmeans.py
```


```shell
# To disable capture of stdout, ```--cpature=no``` or ```-s``` can be used.
uv run pytest --cov -v --capture=no
# show print statement or logging output
uv run pytest --cov -v --capture=sys
# show all
uv run pytest --cov -v --tb=short -s
```