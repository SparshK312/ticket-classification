# IT Ticket Automation Analysis

A Python-based system for analyzing IT ticket data to identify patterns, group similar tickets, and determine automation potential using machine learning and natural language processing techniques.

## Project Overview

This project implements a multi-phase analysis pipeline:

1. **Data Processing**: Load and clean IT ticket data from CSV files
2. **Ticket Grouping**: Use clustering algorithms to group similar tickets
3. **Keyword Extraction**: Extract key terms and analyze automation potential
4. **Analysis & Reporting**: Generate insights and recommendations

## Features

- **Automated Ticket Clustering**: Groups similar tickets using advanced NLP embeddings
- **Keyword Extraction**: Identifies common patterns and technical terms
- **Automation Scoring**: Assesses which ticket types are candidates for automation
- **Comprehensive Reporting**: Generates detailed analysis results
- **Modular Design**: Easy to extend and customize for different use cases

## Project Structure

```
ticket-classification/
├── data/
│   ├── raw/           # Original CSV files
│   └── processed/     # Cleaned and processed data
├── src/
│   ├── __init__.py
│   ├── data_processing.py    # Data loading and preprocessing
│   ├── grouping.py          # Ticket clustering algorithms
│   └── keyword_extraction.py # NLP analysis and automation scoring
├── outputs/           # Analysis results and reports
├── config/           # Configuration files
├── notebooks/        # Jupyter notebooks for exploration
├── main.py          # Main execution script
├── config.py        # Project configuration
└── requirements.txt # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup Instructions

1. **Navigate to project directory**:
   ```bash
   cd "Ticket Classification"
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment**:
   
   **On macOS/Linux**:
   ```bash
   source venv/bin/activate
   ```
   
   **On Windows**:
   ```bash
   venv\Scripts\activate
   ```
   
   You should see `(venv)` in your terminal prompt when the environment is active.

4. **Upgrade pip (recommended)**:
   ```bash
   python -m pip install --upgrade pip
   ```

5. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Virtual Environment Management

**To activate the environment** (do this each time you work on the project):
- macOS/Linux: `source venv/bin/activate`
- Windows: `venv\Scripts\activate`

**To deactivate the environment**:
```bash
deactivate
```

**To verify the environment is active**:
```bash
which python  # Should show path to venv/bin/python
pip list       # Shows installed packages in the virtual environment
```

## Usage

### Basic Usage

Run the complete analysis pipeline:

```bash
python main.py --data-file data/raw/tickets.csv
```

### Configuration

Edit `config.py` to customize:
- Model parameters
- Clustering algorithms
- Output formats
- File paths

### Input Data Format

Your CSV file should contain columns such as:
- `description`: Ticket description/summary
- `category`: Ticket category
- `priority`: Priority level
- `status`: Current status
- `comments`: Additional comments or resolution notes

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black src/
flake8 src/
```

### Jupyter Notebooks

Explore data and experiment with models:

```bash
jupyter notebook notebooks/
```

## Output

The analysis generates:
- **Cluster assignments**: Which tickets belong to which groups
- **Keywords**: Top terms for each cluster
- **Automation scores**: Potential for automation (high/medium/low)
- **Summary reports**: Overview of findings and recommendations

Results are saved in the `outputs/` directory as JSON files and can be easily integrated into dashboards or reporting systems.

## Next Steps

- [ ] Implement web-based dashboard for visualization
- [ ] Add support for additional data formats
- [ ] Integrate with ticketing systems (JIRA, ServiceNow)
- [ ] Develop automated alert system for high-automation-potential tickets
- [ ] Add model performance monitoring and retraining capabilities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.