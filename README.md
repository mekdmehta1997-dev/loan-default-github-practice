# Dummy Loan Default Prediction MLOps

**Practice-only project.** Uses synthetic data, not a production credit-risk model.

## Structure
- `data/`: synthetic loan examples
- `src/`: preprocessing, training and evaluation
- `tests/`: automated tests
- `api/`: FastAPI prediction service
- `.github/workflows/`: CI and manual CD placeholder
- `k8s/`: Kubernetes deployment and service
- `kubeflow/`: independent demonstration Kubeflow training pipeline

## Run locally (PowerShell)
```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.train
uvicorn api.app:app --reload
```
Visit http://localhost:8000/docs.

## Docker-only API run
```powershell
docker build -t loan-default-api:local .
docker run --rm -p 8000:8000 loan-default-api:local
```

## KIND deployment
```powershell
kind load docker-image loan-default-api:local --name loan-cluster
kubectl --context kind-loan-cluster apply -f k8s/
kubectl --context kind-loan-cluster port-forward svc/loan-default-api 8000:8000
```
Change `loan-cluster` and context to your actual KIND cluster. Docker Desktop must be running.

## Kubeflow pipeline compilation
In a separate environment with KFP v2 installed:
```powershell
python -m pip install 'kfp>=2,<3'
python kubeflow/pipeline.py
```
Upload the generated YAML to your Kubeflow Pipelines UI. This is an **independent training demo**; it does not yet pass a trained artifact to the API.

## CI/CD notes
`ci.yml` runs tests, training, evaluation and Docker build on GitHub Actions. `cd.yml` is a manual placeholder and **does not deploy**. Later steps will add GHCR publishing and a secure runner strategy for local KIND.
