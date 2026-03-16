# AI Scheduling

## Usage

This workspace uses Docker Compose to run a persistent Python development container.

### Start container

```bash
docker compose -f docker/compose.yaml up -d
```

### Stop container

```bash
docker compose -f docker/compose.yaml down
```

### Optional: open a shell in the running container

```bash
docker compose -f docker/compose.yaml exec python-app bash
```

### Optional: run the sample app

```bash
python app/main.py
```
