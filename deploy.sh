#!/usr/bin/env bash
# SuperKart - rebuild and verify both containers inside the GitHub Codespace.
# Usage (Codespace terminal, repo root):   bash deploy.sh
set -e
EXPECTED_MD5="a7637e5ab802e60f3bf8b35b9eef87ea"
EXPECTED_SKLEARN="1.6.1"
EXPECTED_XGB="3.4.1"

echo "== 1/5 Sync with GitHub (origin/main) =="
git fetch origin
git reset --hard origin/main
echo "Commit: $(git rev-parse --short HEAD)"

echo "== 2/5 Check model file =="
ACTUAL_MD5=$(md5sum backend/superkart_model.joblib | cut -d' ' -f1)
echo "expected $EXPECTED_MD5"
echo "actual   $ACTUAL_MD5"
if [ "$ACTUAL_MD5" != "$EXPECTED_MD5" ]; then
  echo "ERROR: backend/superkart_model.joblib is not the model trained in the notebook. Re-run the GitHub push cell in Colab, then run this script again."; exit 1
fi

echo "== 3/5 Rebuild containers =="
docker rm -f backend frontend 2>/dev/null || true
docker network create superkart-net 2>/dev/null || true
docker build -t superkart-backend ./backend
docker builder prune -af
docker run -d --name backend --network superkart-net -p 7860:7860 superkart-backend
docker build -t superkart-frontend ./frontend
docker builder prune -af
docker run -d --name frontend --network superkart-net -p 8501:8501 superkart-frontend

echo "== 4/5 Wait for backend =="
for i in $(seq 1 30); do
  if curl -sf http://localhost:7860/health >/dev/null; then break; fi
  sleep 2
done
docker ps --format "table {{.Names}}	{{.Status}}	{{.Ports}}"

echo "== 5/5 Verify deployed model =="
INFO=$(curl -s http://localhost:7860/v1/info)
echo "$INFO"
DEPLOYED_MD5=$(echo "$INFO" | python3 -c "import sys,json; print(json.load(sys.stdin).get('model_md5',''))")
DEPLOYED_SK=$(echo "$INFO" | python3 -c "import sys,json; print(json.load(sys.stdin).get('scikit-learn',''))")
DEPLOYED_XGB=$(echo "$INFO" | python3 -c "import sys,json; print(json.load(sys.stdin).get('xgboost',''))")
if [ "$DEPLOYED_MD5" = "$EXPECTED_MD5" ] && [ "$DEPLOYED_SK" = "$EXPECTED_SKLEARN" ] && [ "$DEPLOYED_XGB" = "$EXPECTED_XGB" ]; then
  echo "DEPLOYMENT OK: model MD5 and library versions match the notebook."
  echo "Now set ports 7860 and 8501 to Public in the PORTS tab, then run the inference cells in Colab."
else
  echo "DEPLOYMENT MISMATCH - run: docker logs backend"; exit 1
fi
