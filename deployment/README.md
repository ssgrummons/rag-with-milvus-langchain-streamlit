# RAG Application Deployment

This directory contains the Docker Compose configuration for deploying the RAG (Retrieval-Augmented Generation) application. The deployment includes the following services:

- Frontend (Streamlit)
- Backend (FastAPI)
- Milvus (Vector Database)
- MinIO (Object Storage)
- etcd (Metadata Storage)
- Attu (Milvus Management UI)

## Prerequisites

- Docker (version 20.10.0 or higher)
- Docker Compose (version 2.0.0 or higher)
- Git

## Directory Structure

```
.
├── docker-compose.yml
├── milvus.yaml
├── vols/
│   ├── etcd/
│   ├── minio/
│   └── milvus/
├── backend/
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── src/
│       └── app.py
└── frontend/
    ├── Dockerfile
    ├── pyproject.toml
    └── src/
        └── app.py
```

## Deployment Steps

1. Clone the repository and navigate to the deployment directory:
```bash
git clone <repository-url>
cd deployment
```

2. Create necessary volume directories:
```bash
mkdir -p vols/{etcd,minio,milvus}
```

3. Build and start all services:
```bash
docker-compose up --build -d
```

The `-d` flag runs the containers in detached mode (background).

## Service Verification

### Check Service Status

1. View all running containers:
```bash
docker-compose ps
```

All services should show "running" status.

2. Check service logs:
```bash
# View logs for all services
docker-compose logs

# View logs for a specific service
docker-compose logs backend
docker-compose logs frontend
docker-compose logs milvus-standalone
docker-compose logs attu
```

### Verify Individual Services

1. Backend API:
```bash
# Test the root endpoint
curl http://localhost:8000/

# Test the health endpoint
curl http://localhost:8000/health
```

Expected responses:
```json
// Root endpoint
{"message": "Hello World from RAG Backend!"}

// Health endpoint
{"status": "healthy"}
```

2. Frontend:
- Open your browser and navigate to: http://localhost:8501
- The Streamlit interface should load

3. Milvus:
```bash
# Check Milvus health
curl http://localhost:19530/healthz
```

4. Attu (Milvus Management UI):
- Open your browser and navigate to: http://localhost:8088
- You can use this interface to:
  - Monitor Milvus collections and indexes
  - View and manage data
  - Execute queries and searches
  - Monitor system performance

## Configuration

The deployment uses the following configuration files:

1. `docker-compose.yml`: Defines all services and their relationships
2. `milvus.yaml`: Contains Milvus-specific configuration
   - No SSL or authentication enabled
   - Local storage paths configured
   - Basic logging settings

## Troubleshooting

1. If services fail to start, check the logs:
```bash
docker-compose logs
```

2. If you need to reset the deployment:
```bash
# Stop all services
docker-compose down

# Remove volume data (warning: this will delete all data)
rm -rf vols/*

# Recreate volumes and start services
mkdir -p vols/{etcd,minio,milvus}
docker-compose up -d
```

3. Common issues:
   - Port conflicts: Ensure ports 8000, 8501, 19530, and 8088 are available
   - Volume permissions: Ensure the `vols` directory has proper permissions
   - Network issues: Check if all services can communicate within the Docker network

## Development

For local development:

1. Stop the containers:
```bash
docker-compose down
```

2. Make changes to the code or configuration

3. Rebuild and restart:
```bash
docker-compose up --build -d
```

## Cleanup

To completely clean up the deployment:

```bash
# Stop and remove containers
docker-compose down

# Remove volume data
rm -rf vols/*

# Remove built images
docker-compose rm -f
docker system prune -f
```

## Production Considerations

1. **Security**
   - Update CORS settings in backend/app.py
   - Set secure passwords for MinIO
   - Configure proper network isolation

2. **Performance**
   - Adjust Milvus configuration based on data size
   - Configure proper resource limits in docker-compose.yml
   - Set up monitoring and alerting

3. **Backup**
   - Regular backup of Milvus data
   - Backup of MinIO storage
   - Backup of etcd data

## Support

For issues or questions:
1. Check the logs using `docker-compose logs`
2. Review the troubleshooting section
3. Check the project documentation
4. Open an issue in the repository 