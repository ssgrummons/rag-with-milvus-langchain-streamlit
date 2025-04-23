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
   The `EMBEDDING_MODEL` environment variable is used in both the runtime and the build environment.  In order to define it once, we will pull it from the `.env` file before we run the Docker Compose command.  This enables us define the environment varialbe once and use it throughout the system.  You can do this by exporting the environment variable:

   ```bash
   export EMBEDDING_MODEL=$(grep EMBEDDING_MODEL ../backend/src/.env | cut -d '=' -f2-) docker-compose -f docker-compose.yml up -d --build
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