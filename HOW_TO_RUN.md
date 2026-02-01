# ScholarGenie - How to Run

## ✅ Pre-Flight Checklist

Before running, ensure you have:
- [x] Python 3.11+ installed ✅ (Detected: Python 3.11.0)
- [ ] PostgreSQL installed (or use Docker)
- [ ] Redis installed (or use Docker)
- [ ] API Keys (OpenAI/Anthropic)
- [ ] 4GB+ RAM available

---

## 🚀 Quick Start (Development Mode)

### Option 1: Run Without Database (Fastest)

**Step 1: Install Dependencies**
```bash
cd C:\Users\aadhi\Desktop\Projects\ScholarGenie
pip install -r requirements.txt
```

**Step 2: Set Environment Variables**
```bash
# Create .env file
copy .env.example .env

# Edit .env with your API keys
notepad .env
```

Add these to `.env`:
```env
# Required API Keys
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
SEMANTIC_SCHOLAR_API_KEY=your-key-here

# JWT Secret (generate random string)
JWT_SECRET_KEY=your-super-secret-key-change-this-in-production

# Redis (optional for dev mode)
REDIS_HOST=localhost
REDIS_PORT=6379

# Database (optional for dev mode)
DATABASE_URL=sqlite:///./scholargenie.db
```

**Step 3: Run API Server**
```bash
# Navigate to project root
cd C:\Users\aadhi\Desktop\Projects\ScholarGenie

# Run FastAPI server
python -m uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

**Step 4: Test the API**

Open browser: **http://localhost:8000/docs**

You'll see the interactive API documentation (Swagger UI).

---

## 🐳 Option 2: Full Production Stack (Docker)

### Prerequisites
```bash
# Install Docker Desktop for Windows
# Download from: https://www.docker.com/products/docker-desktop/
```

### Step 1: Configure Environment
```bash
cd C:\Users\aadhi\Desktop\Projects\ScholarGenie

# Copy environment template
copy .env.example .env

# Edit with your values
notepad .env
```

### Step 2: Start All Services
```bash
# Start PostgreSQL and Redis first
docker-compose -f docker-compose.prod.yml up -d postgres redis

# Wait for databases to be ready (30 seconds)
timeout /t 30

# Run database migrations
docker-compose -f docker-compose.prod.yml run --rm api alembic upgrade head

# Start all services
docker-compose -f docker-compose.prod.yml up -d
```

### Step 3: Verify Services
```bash
# Check all services are running
docker-compose -f docker-compose.prod.yml ps

# Check API health
curl http://localhost/health

# View logs
docker-compose -f docker-compose.prod.yml logs -f api
```

### Step 4: Access Services

| Service | URL | Description |
|---------|-----|-------------|
| **API Docs** | http://localhost/docs | Interactive API documentation |
| **API** | http://localhost/api/ | Main API endpoints |
| **Flower** | http://localhost:5555 | Celery task monitoring |
| **Health** | http://localhost/health | System health check |

---

## 🧪 Testing the API

### 1. Health Check
```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-12-12T10:30:00Z",
  "services": {
    "database": "connected",
    "redis": "connected",
    "agents": {...}
  }
}
```

### 2. Search Papers
```bash
curl -X POST http://localhost:8000/api/papers/search \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"deep learning\", \"max_results\": 5}"
```

### 3. Create Account
```bash
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d "{
    \"email\": \"test@example.com\",
    \"username\": \"testuser\",
    \"password\": \"SecurePass123!\",
    \"full_name\": \"Test User\"
  }"
```

**Save the `access_token` from response!**

### 4. Test CrewAI Multi-Agent (Requires Token)
```bash
# Replace YOUR_TOKEN with the access_token from step 3
curl -X POST http://localhost:8000/api/crew/research \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"topic\": \"transformers in NLP\",
    \"max_papers\": 5,
    \"depth\": \"comprehensive\"
  }"
```

---

## 🔧 Troubleshooting

### Problem: Import Errors

**Solution:**
```bash
# Make sure you're in the project root
cd C:\Users\aadhi\Desktop\Projects\ScholarGenie

# Set PYTHONPATH
set PYTHONPATH=%CD%

# Try running again
python -m uvicorn backend.app:app --reload
```

### Problem: "Module 'crewai' not found"

**Solution:**
```bash
pip install crewai==0.28.8 crewai-tools==0.2.6
pip install langchain==0.1.0 langchain-openai==0.0.2 langchain-anthropic==0.1.1
```

### Problem: "Database connection failed"

**Solution for Development:**
```bash
# Use SQLite instead
# In .env, change DATABASE_URL to:
DATABASE_URL=sqlite:///./scholargenie.db
```

**Solution for Production:**
```bash
# Start PostgreSQL with Docker
docker-compose -f docker-compose.prod.yml up -d postgres

# Wait for it to start
timeout /t 10

# Test connection
docker exec scholargenie-postgres pg_isready -U scholargenie
```

### Problem: "Redis connection failed"

**Solution:**
```bash
# Redis is optional for development
# The app will run without it, just without caching

# To start Redis with Docker:
docker-compose -f docker-compose.prod.yml up -d redis
```

### Problem: Port 8000 already in use

**Solution:**
```bash
# Use a different port
python -m uvicorn backend.app:app --reload --port 8001

# Or find and kill the process using port 8000
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F
```

### Problem: CrewAI execution timeout

**Solution:**
```bash
# CrewAI crews can take 2-5 minutes
# This is normal for multi-agent orchestration
# Be patient or reduce max_papers parameter
```

---

## 📊 Monitoring

### View API Logs
```bash
# Development mode
# Logs appear in terminal where you ran uvicorn

# Docker mode
docker-compose -f docker-compose.prod.yml logs -f api
```

### View Celery Tasks (Docker only)
```bash
# Access Flower UI
# Open browser: http://localhost:5555

# Or view Celery logs
docker-compose -f docker-compose.prod.yml logs -f celery-worker
```

### Check Database
```bash
# Docker mode
docker exec -it scholargenie-postgres psql -U scholargenie

# Then run SQL
\dt  # List tables
SELECT COUNT(*) FROM users;
\q   # Quit
```

---

## 🎯 What Works Right Now

### ✅ Fully Functional
- [x] Paper search (Semantic Scholar)
- [x] Paper summarization
- [x] Knowledge graph construction
- [x] Gap discovery (10 methods)
- [x] Literature review generation
- [x] Grant matching
- [x] Cross-domain knowledge transfer
- [x] **CrewAI multi-agent orchestration** ⭐ NEW
- [x] JWT authentication
- [x] Rate limiting
- [x] Health checks
- [x] API documentation

### ⚠️ Requires API Keys
- [ ] OpenAI API (for GPT-4 crews)
- [ ] Anthropic API (for Claude crews)
- [ ] Semantic Scholar API (optional, works without)

### ⚠️ Requires Services
- [ ] PostgreSQL (optional - use SQLite for dev)
- [ ] Redis (optional - works without, no caching)

---

## 🚦 Development Workflow

### 1. Start Development Server
```bash
# Terminal 1: API Server
cd C:\Users\aadhi\Desktop\Projects\ScholarGenie
set PYTHONPATH=%CD%
python -m uvicorn backend.app:app --reload
```

### 2. Make Changes
```bash
# Edit code in backend/
# Server auto-reloads on changes (--reload flag)
```

### 3. Test Changes
```bash
# Open http://localhost:8000/docs
# Try the endpoint in Swagger UI
```

### 4. Check Logs
```bash
# Logs appear in Terminal 1
# Look for errors or warnings
```

---

## 📦 Production Deployment

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for:
- SSL/TLS setup
- Domain configuration
- Database backups
- Scaling strategies
- Security hardening
- Monitoring setup

---

## 🆘 Quick Help

**Can't install dependencies?**
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Try installing again
pip install -r requirements.txt
```

**Server won't start?**
```bash
# Check Python version (need 3.11+)
python --version

# Check if port is free
netstat -ano | findstr :8000

# Try different port
python -m uvicorn backend.app:app --reload --port 8001
```

**CrewAI not working?**
```bash
# Verify API keys in .env
# Make sure you have either OPENAI_API_KEY or ANTHROPIC_API_KEY

# Test manually
python -c "import crewai; print('CrewAI OK')"
python -c "import langchain_anthropic; print('LangChain OK')"
```

---

## ✅ Success Criteria

You know it's working when:

1. **Server starts** ✓
   ```
   INFO:     Uvicorn running on http://0.0.0.0:8000
   INFO:     Application startup complete
   ```

2. **/health returns healthy** ✓
   ```json
   {"status": "healthy", "services": {...}}
   ```

3. **Swagger UI loads** ✓
   - Open http://localhost:8000/docs
   - See 90+ endpoints

4. **Can search papers** ✓
   - POST /api/papers/search works
   - Returns paper results

5. **CrewAI crews work** ✓
   - POST /api/crew/research returns results
   - Takes 1-5 minutes (normal)

---

## 🎉 You're Ready!

Once the server is running and /health returns "healthy":

1. **Explore API**: http://localhost:8000/docs
2. **Read docs**: [PRODUCTION_README.md](PRODUCTION_README.md)
3. **Try CrewAI**: [CREWAI_IMPLEMENTATION.md](CREWAI_IMPLEMENTATION.md)
4. **Deploy**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

---

## 📞 Support

**Common Issues?** Check [TROUBLESHOOTING] section above

**More Questions?**
- Check documentation files in project root
- Review error messages carefully
- Ensure all API keys are set correctly

**Ready to Deploy?**
- Read [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- Follow security checklist
- Setup monitoring

---

**Current Status**: ✅ **READY TO RUN**

All features implemented, all dependencies listed, all docs created.
Just need API keys and you're good to go! 🚀
