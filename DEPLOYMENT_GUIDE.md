# AI Architectural Space Analyzer - Complete Deployment Guide

## Overview
This is a comprehensive Streamlit-based application for analyzing architectural drawings (DWG/DXF files) with AI-powered room classification and intelligent furniture placement optimization.

## Key Features
- **Native DWG/DXF Support**: Advanced parsing with multiple fallback methods
- **Intelligent Zone Detection**: Automatic room identification from CAD drawings
- **AI Room Classification**: Smart room type detection with confidence scoring
- **Advanced Placement Optimization**: Furniture/equipment placement with multiple strategies
- **Professional Export**: CAD-compatible DXF, SVG, PDF exports
- **Database Integration**: PostgreSQL support for project management
- **Collaborative Features**: Real-time commenting and version control

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd dwg-analyzer

# Install dependencies using UV (recommended)
uv sync

# Or use pip
pip install -r requirements_deploy.txt
```

### 2. Database Configuration

The application supports both PostgreSQL (production) and SQLite (development).

#### PostgreSQL Setup:
```bash
export DATABASE_URL="postgresql://username:password@host:port/database"
```

#### For this project specifically:
```bash
export DATABASE_URL="postgresql://yang:nNTm6Q4un1aF25fmVvl7YqSzWffyznIe@dpg-d0t3rlili9vc739k84gg-a.oregon-postgres.render.com/dg4u_tiktok_bot"
```

### 3. Run Locally

```bash
streamlit run app.py --server.port 5000 --server.address 0.0.0.0
```

## Deployment Options

### Option 1: Streamlit Cloud Deployment

1. **Prepare Repository**:
   - Ensure all files are in GitHub repository
   - Create `requirements.txt` from `requirements_deploy.txt`
   - Add `.streamlit/config.toml` with server settings

2. **Deploy to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select branch (usually `main`)
   - Set main file path: `app.py`

3. **Configure Secrets**:
   Add these to Streamlit Cloud secrets:
   ```toml
   [secrets]
   DATABASE_URL = "postgresql://yang:nNTm6Q4un1aF25fmVvl7YqSzWffyznIe@dpg-d0t3rlili9vc739k84gg-a.oregon-postgres.render.com/dg4u_tiktok_bot"
   
   # Optional: AI API keys for enhanced features
   GEMINI_API_KEY = "your-gemini-api-key"
   OPENAI_API_KEY = "your-openai-api-key"
   ```

4. **Streamlit Cloud Files Structure**:
   ```
   your-repo/
   ├── app.py
   ├── requirements.txt
   ├── .streamlit/
   │   └── config.toml
   ├── src/
   │   ├── __init__.py
   │   ├── dwg_parser.py
   │   ├── enhanced_zone_detector.py
   │   ├── navigation_manager.py
   │   ├── placement_optimizer.py
   │   └── ... (all other modules)
   └── README.md
   ```

### Option 2: Render.com Deployment

1. **Prepare for Render**:
   Create `render.yaml`:
   ```yaml
   services:
     - type: web
       name: dwg-analyzer
       env: python
       buildCommand: "pip install -r requirements_deploy.txt"
       startCommand: "streamlit run app.py --server.port $PORT --server.address 0.0.0.0"
       envVars:
         - key: DATABASE_URL
           value: "postgresql://yang:nNTm6Q4un1aF25fmVvl7YqSzWffyznIe@dpg-d0t3rlili9vc739k84gg-a.oregon-postgres.render.com/dg4u_tiktok_bot"
   ```

2. **Deploy to Render**:
   - Connect GitHub repository to Render
   - Select "Web Service"
   - Configure build and start commands
   - Add environment variables

3. **Environment Variables for Render**:
   ```
   DATABASE_URL=postgresql://yang:nNTm6Q4un1aF25fmVvl7YqSzWffyznIe@dpg-d0t3rlili9vc739k84gg-a.oregon-postgres.render.com/dg4u_tiktok_bot
   PYTHON_VERSION=3.11
   ```

### Option 3: Docker Deployment

1. **Create Dockerfile**:
   ```dockerfile
   FROM python:3.11-slim

   WORKDIR /app

   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       gcc \
       g++ \
       && rm -rf /var/lib/apt/lists/*

   # Copy requirements and install Python dependencies
   COPY requirements_deploy.txt .
   RUN pip install --no-cache-dir -r requirements_deploy.txt

   # Copy application code
   COPY . .

   # Expose port
   EXPOSE 5000

   # Health check
   HEALTHCHECK CMD curl --fail http://localhost:5000/_stcore/health

   # Run the application
   CMD ["streamlit", "run", "app.py", "--server.port=5000", "--server.address=0.0.0.0"]
   ```

2. **Build and Run**:
   ```bash
   docker build -t dwg-analyzer .
   docker run -p 5000:5000 -e DATABASE_URL="your-database-url" dwg-analyzer
   ```

## File Structure

```
dwg-analyzer/
├── app.py                          # Main application entry point
├── requirements_deploy.txt         # Production dependencies
├── replit.md                      # Project documentation
├── .streamlit/
│   └── config.toml                # Streamlit configuration
├── src/
│   ├── __init__.py
│   ├── dwg_parser.py              # DWG/DXF file parsing
│   ├── enhanced_zone_detector.py   # Advanced zone detection
│   ├── navigation_manager.py       # Navigation and workflow
│   ├── placement_optimizer.py      # Furniture placement optimization
│   ├── ai_analyzer.py             # AI room classification
│   ├── ai_integration.py          # Multi-AI service integration
│   ├── visualization.py           # Interactive visualizations
│   ├── database.py                # Database models and operations
│   ├── export_utils.py            # Export functionality
│   ├── cad_export.py              # CAD export capabilities
│   ├── bim_integration.py         # BIM compliance features
│   ├── furniture_catalog.py       # Furniture catalog management
│   ├── collaborative_features.py  # Real-time collaboration
│   └── multi_floor_analysis.py    # Multi-floor building analysis
├── attached_assets/               # Sample files and uploads
└── sample_files/                  # Sample DWG/DXF files
```

## Configuration Files

### .streamlit/config.toml
```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
maxUploadSize = 200

[theme]
base = "light"
primaryColor = "#1f77b4"
```

### requirements_deploy.txt
```
streamlit>=1.28.0
ezdxf>=1.0.0
dxfgrabber>=1.0.1
shapely>=2.0.0
matplotlib>=3.7.0
plotly>=5.15.0
pandas>=2.0.0
numpy>=1.24.0
reportlab>=4.0.0
rectpack>=0.2.0
opencv-python-headless>=4.8.0
scikit-learn>=1.3.0
scipy>=1.11.0
psycopg2-binary>=2.9.0
sqlalchemy>=2.0.0
google-generativeai>=0.8.0
pydantic>=2.0.0
networkx>=3.0.0
deap>=1.4.0
pymupdf>=1.26.0
```

## Application Usage Guide

### 1. File Upload and Analysis
- Upload DWG or DXF files (up to 200MB)
- Application automatically detects room zones
- Processes CAD entities and extracts geometric data
- Provides detailed parsing feedback

### 2. Room Detection and Classification
- Identifies closed polygons as room zones
- Uses line network analysis for complex drawings
- AI-powered room type classification
- Confidence scoring for each detection

### 3. Furniture/Equipment Placement
- Multiple placement strategies:
  - Center placement for large items (kitchen islands)
  - Wall-adjacent for workstations
  - Corner placement for equipment racks
  - Grid-based optimization
- Intelligent spacing and clearance calculation
- Rotation optimization when beneficial
- Accessibility scoring for placements

### 4. Visualization and Export
- Interactive 2D/3D visualizations
- Professional CAD export (DXF format)
- PDF reports with statistics
- SVG export for web use
- Comprehensive analysis reports

### 5. Advanced Features
- Multi-floor building analysis
- BIM compliance checking (IFC, COBie)
- Collaborative commenting
- Project version control
- Database integration for project management

## Performance Optimization

### For Large Files:
- File size limit: 200MB
- Processing optimization for files with 100k+ entities
- Memory-efficient parsing strategies
- Progressive loading for complex drawings

### Database Performance:
- Connection pooling for PostgreSQL
- Indexed queries for fast retrieval
- Automatic fallback to SQLite for development
- Regular cleanup of temporary data

## Troubleshooting

### Common Issues:

1. **DWG File Parsing Errors**:
   - Convert to DXF using AutoCAD, FreeCAD, or LibreCAD
   - Ensure file is not password-protected
   - Check file format version compatibility

2. **Memory Issues with Large Files**:
   - Increase available memory for deployment
   - Use file compression
   - Consider file splitting for very large drawings

3. **Database Connection Errors**:
   - Verify DATABASE_URL environment variable
   - Check network connectivity to database
   - Ensure database credentials are correct

4. **Performance Issues**:
   - Enable caching in Streamlit
   - Optimize zone detection parameters
   - Consider using CDN for static assets

## Security Considerations

- Environment variables for sensitive data
- Database connection encryption
- File upload validation and sanitization
- No storage of user files beyond session
- Secure handling of API keys

## Monitoring and Logging

- Application health checks
- Error logging and reporting
- Performance metrics tracking
- Database query monitoring
- User activity analytics

## Support and Maintenance

### Regular Maintenance:
- Database cleanup and optimization
- Dependency updates
- Security patches
- Performance monitoring

### Backup Strategy:
- Database regular backups
- Configuration backup
- Application code versioning
- User data protection

## API Integration

The application supports multiple AI services:
- Google Gemini AI (primary)
- OpenAI GPT models
- Custom AI endpoints
- Fallback to geometric analysis

Configure API keys as environment variables:
```bash
export GEMINI_API_KEY="your-gemini-key"
export OPENAI_API_KEY="your-openai-key"
```

## Scaling Considerations

### Horizontal Scaling:
- Stateless application design
- Database connection pooling
- Load balancer configuration
- Session state management

### Vertical Scaling:
- Memory optimization for large files
- CPU optimization for analysis algorithms
- Storage optimization for temporary files
- Network optimization for file uploads

---

This deployment guide provides comprehensive instructions for deploying the AI Architectural Space Analyzer in various environments. Choose the deployment option that best fits your infrastructure and requirements.