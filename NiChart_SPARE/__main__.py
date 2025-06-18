#!/usr/bin/env python3
"""
NiChart_SPARE - Main entry point for SPARE scores calculation

This script provides command-line interface for training and inference of SPARE models.
Supported SPARE types: BA (Brain Age), AD (Alzheimer's), HT (Hypertension), HL (Hyperlipidemia), 
T2B (Diabetes), SM (Smoking), OB (Obesity)
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Import pipeline modules
from .pipelines import spare_ad, spare_ba, spare_ht

def setup_logging(verbose=False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def validate_file_path(file_path, description):
    """Validate that a file path exists"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{description} file not found: {file_path}")
    return file_path

def validate_directory_path(dir_path, description):
    """Validate that a directory path exists and is writable"""
    dir_path = Path(dir_path)
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
    if not os.access(dir_path, os.W_OK):
        raise PermissionError(f"Cannot write to {description} directory: {dir_path}")
    return str(dir_path)

def get_pipeline_module(spare_type):
    """Get the appropriate pipeline module based on SPARE type"""
    spare_type = spare_type.upper()
    
    pipeline_map = {
        'AD': spare_ad,
        'BA': spare_ba,
        'HT': spare_ht,
        # Add other SPARE types as they become available
        # 'HL': spare_hl,
        # 'T2B': spare_t2b,
        # 'SM': spare_sm,
        # 'OB': spare_ob,
    }
    
    if spare_type not in pipeline_map:
        raise ValueError(f"Unsupported SPARE type: {spare_type}. "
                        f"Supported types: {list(pipeline_map.keys())}")
    
    return pipeline_map[spare_type]

def main():
    """Main entry point for NiChart_SPARE"""
    parser = argparse.ArgumentParser(
        description="NiChart_SPARE - SPARE scores calculation from Brain ROI Volumes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training
  NiChart_SPARE -a trainer -t AD -i /path/to/input_file.csv -mo /path/to/model.pkl.gz -v True
  
  # Inference
  NiChart_SPARE -a inference -t AD -i /path/to/test_file.csv -mo /path/to/model.pkl.gz -v False -o /path/to/output.csv
        """
    )
    
    # Required arguments
    parser.add_argument('-a', '--action', 
                       required=True,
                       choices=['trainer', 'inference'],
                       help='Action to perform: trainer (training) or inference (prediction)')
    
    parser.add_argument('-t', '--type',
                       required=True,
                       help='SPARE type: AD, BA, HT, HL, T2B, SM, OB')
    
    parser.add_argument('-i', '--input',
                       required=True,
                       help='Input file path (CSV format)')
    
    parser.add_argument('-mo', '--model',
                       required=True,
                       help='Model file path (for training: output path, for inference: input path)')
    
    # Optional arguments
    parser.add_argument('-v', '--verbose',
                       type=str,
                       default='False',
                       choices=['True', 'False'],
                       help='Verbose output (True/False)')
    
    parser.add_argument('-o', '--output',
                       help='Output file path (required for inference)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    verbose = args.verbose.lower() == 'true'
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Validate input file
        input_file = validate_file_path(args.input, "Input")
        logger.info(f"Input file: {input_file}")
        
        # Validate model path
        if args.action == 'trainer':
            # For training, ensure the directory exists
            model_dir = os.path.dirname(args.model)
            if model_dir:
                validate_directory_path(model_dir, "Model output")
        else:
            # For inference, ensure the model file exists
            validate_file_path(args.model, "Model")
        
        logger.info(f"Model path: {args.model}")
        
        # Validate output file for inference
        if args.action == 'inference':
            if not args.output:
                raise ValueError("Output file path (-o) is required for inference")
            output_dir = os.path.dirname(args.output)
            if output_dir:
                validate_directory_path(output_dir, "Output")
            logger.info(f"Output file: {args.output}")
        
        # Get appropriate pipeline module
        pipeline_module = get_pipeline_module(args.type)
        logger.info(f"Using pipeline for SPARE type: {args.type}")
        
        # Execute the appropriate action
        if args.action == 'trainer':
            logger.info("Starting training...")
            # Call training function from pipeline module
            # This would be implemented in the respective pipeline modules
            if hasattr(pipeline_module, 'train'):
                pipeline_module.train(
                    input_file=input_file,
                    model_path=args.model,
                    verbose=verbose
                )
            else:
                raise NotImplementedError(f"Training not implemented for SPARE type: {args.type}")
                
        elif args.action == 'inference':
            logger.info("Starting inference...")
            # Call inference function from pipeline module
            # This would be implemented in the respective pipeline modules
            if hasattr(pipeline_module, 'predict'):
                pipeline_module.predict(
                    input_file=input_file,
                    model_path=args.model,
                    output_file=args.output,
                    verbose=verbose
                )
            else:
                raise NotImplementedError(f"Inference not implemented for SPARE type: {args.type}")
        
        logger.info("Operation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
