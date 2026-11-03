#!/usr/bin/env bash
# ./scripts/cold_start.sh native|docker [N]
MODE=${1:-docker}; RUNS=${2:-5}
export PYTHONPATH=. MODEL_PATH=artifacts/model.pt

if ss -ltn | grep -q ':8000 '; then
  echo "Порт 8000 зайнятий, зупини інший сервер/контейнер" >&2; exit 1
fi

for i in $(seq "$RUNS"); do
  start=$(date +%s.%N)
  if [ "$MODE" = native ]; then
    uvicorn app.api:app --port 8000 >/dev/null 2>&1 &
    PID=$!
  else
    docker run -d --rm --name kws-cold -p 8000:8000 kws-api:1.0.0 >/dev/null || exit 1
  fi
  until curl -fs localhost:8000/health >/dev/null; do sleep 0.05; done
  h=$(date +%s.%N)
  curl -fs -X POST -F "file=@samples/yes.wav" localhost:8000/predict >/dev/null
  p=$(date +%s.%N)
  awk -v s="$start" -v h="$h" -v p="$p" -v i="$i" \
    'BEGIN{printf "run %d: до /health = %.2f с, до першого /predict = %.2f с\n", i, h-s, p-s}'
  if [ "$MODE" = native ]; then kill $PID; wait $PID 2>/dev/null
  else docker stop -t 1 kws-cold >/dev/null; fi
  sleep 1
done