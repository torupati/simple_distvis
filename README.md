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

```shell
gcloud builds submit --config cloudbuild.yaml
```
