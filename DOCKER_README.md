# Docker Setup for CNN Bio Project

## Prerequisites
- Docker installed on your system
- Docker Compose installed

## Quick Start

### 1. Build the Docker image
```bash
docker-compose build
```

### 2. Run the container
```bash
docker-compose up -d
```

### 3. Execute the menu script
```bash
docker-compose exec cnn-bio ./menu.sh
```

Or run Python scripts directly:
```bash
docker-compose exec cnn-bio python CNN_pytorch.py
docker-compose exec cnn-bio python CNN_tensorflow.py
```

## Directory Structure

When using Docker, the following directories are created and managed:

- **`./datasets`** - Mount point for your dataset files
  - Place your dataset files here (e.g., `./datasets/Segmentazione`)
  - Accessible inside container at `/app/datasets`

- **`./results`** - Mount point for output/results
  - Model outputs and results are saved here
  - Accessible inside container at `/app/results`

- **`.`** (project root) - Your entire project is mounted
  - Changes made locally are reflected in the container
  - Changes made in container are reflectedlocally

## Usage Examples

### Run the interactive menu
```bash
docker-compose up -d
docker-compose exec cnn-bio bash
# Inside container:
./menu.sh
```

### Run PyTorch implementation
```bash
docker-compose exec cnn-bio python CNN_pytorch.py
```

### Run TensorFlow implementation
```bash
docker-compose exec cnn-bio python CNN_tensorflow.py
```

### View running containers
```bash
docker-compose ps
```

### Stop containers
```bash
docker-compose down
```

### View logs
```bash
docker-compose logs -f cnn-bio
```

## Preparing Datasets

1. Create a `datasets` directory in your project root
2. Copy your dataset files into `./datasets/`
3. Inside the container, datasets are available at `/app/datasets`

Example structure:
```
./datasets/
├── Segmentazione/
│   ├── images/
│   ├── masks/
│   └── ...
└── other_dataset/
```

## Environment Variables

The container sets:
- `DATASET_PATH=/app/datasets` - Default dataset path
- `PYTHONUNBUFFERED=1` - Python output buffering disabled

To add more environment variables, edit `docker-compose.yml` under the `environment` section.

## Customization

### Modify resource limits
Uncomment the `deploy` section in `docker-compose.yml` to set CPU and memory limits:
```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
```

### Use GPU support
To enable GPU support, add to the `cnn-bio` service in `docker-compose.yml`:
```yaml
runtime: nvidia
```

And ensure NVIDIA Docker runtime is installed on your system.

## Troubleshooting

### Container fails to start
Check logs:
```bash
docker-compose logs cnn-bio
```

### Permission denied errors
Ensure dataset files have proper permissions. You may need to run:
```bash
chmod -R 755 ./datasets
```

### Out of memory
Adjust memory limits in `docker-compose.yml` or use the `--memory` flag:
```bash
docker-compose up -d --memory 8g
```
