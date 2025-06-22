# The DWG Analyzer Pro V 1.0

## AI-Powered Architectural Space Analyzer

A comprehensive professional solution for architectural analysis and space planning that combines advanced AI algorithms with building information modeling (BIM) to provide intelligent room detection, optimal furniture placement, and complete analysis reporting.

### Features

#### Standard Mode
- **DWG/DXF File Support**: Parse and analyze architectural drawings
- **AI Room Classification**: Intelligent room type detection with confidence scoring
- **Optimal Box Placement**: Advanced algorithms for furniture/equipment placement
- **Interactive Visualization**: 2D/3D plan visualization with real-time interaction
- **Statistical Analysis**: Comprehensive space utilization and efficiency metrics
- **Export Capabilities**: PDF reports, CSV data, and JSON analysis export

#### Advanced Professional Mode
- **Gemini AI Integration**: 95%+ accuracy room classification using Google's Gemini AI
- **BIM Integration**: Full IFC/COBie compliance and standards validation
- **Multi-Floor Analysis**: Complete building analysis across multiple floors
- **Team Collaboration**: Real-time collaborative planning and commenting
- **Professional Furniture Catalog**: Integration with pricing and procurement data
- **Advanced Optimization**: Genetic algorithms and simulated annealing
- **CAD Export Package**: Professional drawing packages (DXF, SVG, 3D models)
- **Database Integration**: PostgreSQL for project management and history

### Technology Stack

- **Frontend**: Streamlit with responsive design
- **AI/ML**: Google Gemini AI, scikit-learn, numpy
- **CAD Processing**: ezdxf for DWG/DXF parsing
- **Visualization**: Plotly for interactive charts and 3D visualization
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Export**: ReportLab for PDF generation, matplotlib for charts
- **Optimization**: scipy, rectpack for advanced algorithms

### Installation

1. Clone the repository:
```bash
git clone https://github.com/rehmanul/The-DWG-Analyzer-Pro-V-1.0.git
cd The-DWG-Analyzer-Pro-V-1.0
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export GEMINI_API_KEY="your_gemini_api_key"
export DATABASE_URL="your_postgresql_url"
```

4. Run the application:
```bash
streamlit run app.py --server.port 5000
```

### Usage

1. **Upload DWG/DXF Files**: Support for single or multiple architectural files
2. **Choose Analysis Mode**: Standard for basic analysis, Advanced for professional workflows
3. **Configure Parameters**: Set box dimensions, margins, and optimization settings
4. **Run AI Analysis**: Get intelligent room classification and optimal placement
5. **Review Results**: Interactive visualization, detailed statistics, and insights
6. **Export Reports**: Download comprehensive analysis packages

### API Keys Required

- **Gemini AI**: Get your free API key from [Google AI Studio](https://makersuite.google.com)
- **Database**: PostgreSQL connection string for data persistence

### File Support

- **Input Formats**: DWG, DXF (AutoCAD 2018 or later recommended)
- **Export Formats**: PDF, CSV, JSON, DXF, SVG, IFC
- **Maximum File Size**: 50MB per file

### Advanced Features

#### BIM Integration
- IFC 4.3 compliance validation
- COBie 2.4 data structuring
- Standards compliance scoring
- 3D model generation

#### Collaboration
- Real-time team planning
- Comment and annotation system
- Project sharing and permissions
- Version control and history

#### Optimization Algorithms
- Genetic algorithm optimization
- Simulated annealing
- Multi-objective optimization
- Space utilization maximization

### Database Schema

The application uses PostgreSQL with the following main tables:
- `projects`: Project management and metadata
- `dwg_files`: File information and parsing results
- `zones`: Individual room/space data
- `analyses`: Analysis results and parameters
- `furniture_placements`: Placement optimization results
- `bim_models`: BIM integration data
- `collaborators`: Team collaboration features

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### License

This project is licensed under the MIT License - see the LICENSE file for details.

### Support

For support, feature requests, or bug reports, please open an issue on GitHub.

### Version History

- **v1.0.0**: Initial release with full AI integration and BIM support
  - Gemini AI integration
  - PostgreSQL database
  - Advanced optimization algorithms
  - Professional export capabilities
  - Responsive web interface

---

Built with ❤️ using Streamlit and Google Gemini AI