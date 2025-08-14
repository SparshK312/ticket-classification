#!/usr/bin/env python3
"""
IT Ticket Automation Analysis - Main Entry Point

This module serves as the main entry point for the IT ticket analysis system.
It orchestrates the data processing, grouping, and analysis pipeline.
"""

import logging
from pathlib import Path
from typing import Optional

from src.data_processing import DataProcessor
from src.grouping import TicketGrouper
from src.keyword_extraction import KeywordExtractor
from config import Config


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('outputs/ticket_analysis.log'),
            logging.StreamHandler()
        ]
    )


def main(data_file: Optional[str] = None) -> None:
    """
    Main pipeline for IT ticket analysis.
    
    Args:
        data_file: Path to the CSV file containing ticket data.
                  If None, uses default from config.
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting IT Ticket Analysis Pipeline")
    
    try:
        # Initialize components
        config = Config()
        processor = DataProcessor(config)
        grouper = TicketGrouper(config)
        extractor = KeywordExtractor(config)
        
        # Load and process data
        data_path = data_file or config.DEFAULT_DATA_FILE
        logger.info(f"Loading data from: {data_path}")
        
        df = processor.load_data(data_path)
        df_processed = processor.preprocess_data(df)
        
        # Group similar tickets
        logger.info("Grouping similar tickets...")
        groups = grouper.group_tickets(df_processed)
        
        # Extract keywords and automation potential
        logger.info("Extracting keywords and analyzing automation potential...")
        results = extractor.analyze_groups(groups, df_processed)
        
        # Save results
        output_path = Path(config.OUTPUT_DIR) / "analysis_results.json"
        extractor.save_results(results, output_path)
        
        logger.info(f"Analysis complete. Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="IT Ticket Automation Analysis")
    parser.add_argument(
        "--data-file", 
        type=str, 
        help="Path to CSV file containing ticket data"
    )
    
    args = parser.parse_args()
    main(args.data_file)