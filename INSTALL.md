# Installation Guide

## Quick Setup

1. **Clone the repository:**
```bash
git clone https://github.com/rehmanul/The-DWG-Analyzer-Pro-V-1.0.git
cd The-DWG-Analyzer-Pro-V-1.0
```

2. **Install Python dependencies:**
```bash
pip install streamlit ezdxf shapely matplotlib plotly pandas numpy reportlab rectpack opencv-python scikit-learn scipy psycopg2-binary sqlalchemy google-genai pydantic
```

3. **Set up API keys:**
```bash
# Get your Gemini AI API key from https://makersuite.google.com
export GEMINI_API_KEY="your_gemini_api_key_here"

# Optional: Set up PostgreSQL database
export DATABASE_URL="postgresql://username:password@localhost:5432/dwg_analyzer"
```

4. **Run the application:**
```bash
streamlit run app.py --server.port 5000
```

5. **Access the application:**
Open your browser and go to `http://localhost:5000`

## Production Deployment

For production deployment on platforms like Replit, Heroku, or cloud services:

1. Add environment variables in your platform's settings
2. Ensure PostgreSQL database is configured
3. Set server configuration for your platform
4. Deploy with the included configuration files

## API Keys Required

- **Gemini AI**: Required for advanced AI features
- **Database URL**: Optional for data persistence (uses in-memory storage if not provided)