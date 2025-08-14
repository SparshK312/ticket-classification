# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Activate virtual environment (required for all operations)
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Verify environment
which python  # Should show venv/bin/python
```

### Key Analysis Scripts
```bash
# Complete ticket processing pipeline (6,964 → 3,847 → classifications)
python detect_identical_tickets.py           # Create fuzzy groups (411 groups from 3,528 tickets)
python create_consolidated_dataset.py        # Consolidate to 3,847 tickets
python improved_semantic_grouping_clean.py   # Hardcoded keyword classification (639 tickets)
python hierarchical_clustering.py            # Cluster remaining 3,208 tickets
python final_ticket_consolidation.py         # Complete end-to-end classification

# Extract results for analysis
python extract_classified_tickets.py         # Export 639 hardcoded classifications
python extract_hierarchical_tickets.py       # Export 3,208 hierarchical classifications

# Individual analysis components
python semantic_analysis.py                  # Semantic embeddings + DBSCAN (experimental)
python cluster_experiments.py                # Systematic clustering comparison
python dynamic_clustering.py                 # Dynamic cluster determination
```

### LLM Automation Analysis (Plan2 - NEW)
```bash
# Prerequisites check and setup
python check_dependencies.py                 # Verify Ollama, packages, and data files
ollama serve                                  # Ensure Ollama is running (separate terminal)
ollama list                                   # Verify llama3.1:8b model is available

# LLM automation classification pipeline  
python test_automation_classifier.py         # Test setup with sample groups (run first!)
python llm_automation_classifier.py          # Analyze all 209 problem groups for automation potential

# Results will be in outputs/ directory:
# - automation_analysis_results_TIMESTAMP.csv     # Main classification results
# - detailed_automation_analysis_TIMESTAMP.json   # Complete analysis with reasoning
# - automation_summary_report_TIMESTAMP.txt       # Business intelligence summary
```

### Testing and Code Quality
```bash
# Code formatting
black src/
flake8 src/

# Run tests
pytest

# Jupyter notebooks for exploration
jupyter notebook notebooks/
```

## Architecture Overview

### Multi-Stage Ticket Classification Pipeline

This system implements a sophisticated **6-stage ticket classification pipeline** that processes enterprise IT tickets through multiple techniques:

**Stage 1: Identical Detection** (`detect_identical_tickets.py`)
- Uses fuzzy string matching with 92% similarity threshold
- Normalizes variable elements (store numbers, names, codes)
- Creates representative tickets for identical groups
- **Input**: 6,964 raw tickets → **Output**: 411 groups representing 3,528 similar tickets

**Stage 2: Consolidation** (`create_consolidated_dataset.py`)
- Combines 411 representatives + 3,436 unique tickets = 3,847 total
- Preserves original ticket relationships and metadata
- Creates analysis-ready dataset

**Stage 3: Hardcoded Classification** (`improved_semantic_grouping_clean.py`)
- Rule-based classification using required + supporting keywords
- Confidence scoring system (high/medium/low)
- **Pattern Examples**: 
  - Vision orders: "order" + "amend"/"locked"/"unable"
  - Till connectivity: "till" + "server"/"offline"/"network"
- **Result**: 639 tickets classified into 7 specific patterns

**Stage 4: Hierarchical Clustering** (`hierarchical_clustering.py`)
- Two-level clustering system for remaining 3,208 tickets
- **Level 1**: 10 business categories using Agglomerative clustering
- **Level 2**: HDBSCAN sub-clustering for large groups (>10% threshold)
- Edge case handling for complex tickets

**Stage 5: Business Mapping** (`business_clustering_structure.py`)
- Maps technical clusters to business categories
- Defines routing queues, SLA hours, priority levels
- **Special case**: Printing services split into 3 specialized queues

**Stage 6: Final Consolidation** (`final_ticket_consolidation.py`)
- Traces all 6,964 original tickets through complete pipeline
- Inheritance logic for group member classifications
- 100% classification coverage with confidence tracking

### Key Data Flows

**Embeddings Pipeline**: Uses SentenceTransformer `all-MiniLM-L6-v2` model with L2 normalization and cosine distance calculations.

**Classification Hierarchy**: 
- 639 tickets → 7 hardcoded patterns (2 categories: Till, Vision)
- 3,208 tickets → 10 business categories + specialized sub-routing
- Total: ~13 distinct problem categories with production routing

**File Outputs**:
- `outputs/improved_classification_results.json` - Hardcoded classification results
- `outputs/hierarchical_clustering_results.json` - Hierarchical analysis
- `outputs/final_complete_ticket_classification.csv` - Complete classification
- `outputs/*_tickets.csv` - Extracted datasets for review

### Technical Implementation

**NLP Stack**: sentence-transformers, scikit-learn, HDBSCAN, NLTK
**Clustering Methods**: DBSCAN (failed), Agglomerative (successful), HDBSCAN (sub-clustering)
**Distance Metrics**: Cosine distance with normalized embeddings
**Validation**: Davies-Bouldin, Silhouette, Calinski-Harabasz scores

### Configuration

Primary configuration in `config.py` with settings for:
- Embedding models and clustering parameters
- Text preprocessing and keyword extraction
- Output formats and logging levels
- Performance tuning (multiprocessing, caching)

### Data Structure

```
data/processed/
├── clean_tickets.csv           # Original 6,964 cleaned tickets
├── consolidated_tickets.csv    # 3,847 consolidated tickets
└── identical_ticket_groups.csv # Fuzzy matching groups

outputs/
├── *_classification_results.json    # Classification outputs
├── *_clustered_tickets.csv         # Extracted classifications
└── final_complete_ticket_classification.csv  # Complete results
```

## Important Notes

- **Virtual environment is mandatory** - All scripts depend on specific ML library versions
- **Order dependency**: Scripts must run in sequence for complete pipeline
- **McAfee compatibility**: scipy extensions may be quarantined; add project to exclusions
- **Memory requirements**: Embedding generation requires ~4GB RAM for full dataset
- **Output validation**: Each stage includes ticket count validation and coverage verification