#!/usr/bin/env python3
"""
NiChart_SPARE - Main entry point for SPARE scores calculation

This script provides command-line interface for training and inference of SPARE models.
Supported SPARE types: 
    - BA (Brain Age)
    - AD (Alzheimer's)
    - CVMs: HT (Hypertension), HL (Hyperlipidemia), T2B (Diabetes), SM (Smoking), OB (Obesity)
"""

import argparse
import sys
# import pandas as pd
# from .pipelines import spare_ad, spare_ba, spare_ht
from .svm import train_svm_model, predict_svm_model

# Entry point & CLI Args
def main():
    """Main entry point for NiChart_SPARE"""
    parser = argparse.ArgumentParser(
        description="NiChart_SPARE - SPARE scores calculation from Brain ROI Volumes",
        epilog="""
            Examples:
            # Train AD model with hyperparameter tuning
            NiChart_SPARE -a trainer -t AD -i data.csv -mo model.pkl -v True
            
            # Train final model after tuning
            NiChart_SPARE -a trainer -t AD -i data.csv -mo model.pkl -v True -f True
            
            # Make predictions
            NiChart_SPARE -a inference -t AD -i test.csv -m model.pkl -o predictions.csv
        """
    )
    
    # Required arguments
    parser.add_argument('-a', '--action', required=True, choices=['trainer', 'inference'],
                       help='Action to perform: trainer or inference')
    parser.add_argument('-t', '--type', required=True, 
                       help='SPARE type: BA (Brain Age), AD (Alzheimer\'s), HT (Hypertension)')
    parser.add_argument('-i', '--input', required=True,
                       help='Input CSV file path')
    # Train/Test specifci arguments
    parser.add_argument('-f', '--final', type=str, default='False',
                       help='Train final model on entire dataset (True/False)')
    parser.add_argument('-mo', '--model_output', 
                       help='Output model file path (for training)')
    parser.add_argument('-m', '--model', 
                       help='Input model file path (for inference)')
    parser.add_argument('-o', '--output', 
                       help='Output CSV file path (for inference)')
    # data preprocessing arguments
    parser.add_argument('-kv', '--key_variable',
                       help='Name of column indicating unique data points in the input CSV')
    parser.add_argument('-tc', '--target_column', default='target',
                       help='Name of target column in CSV')
    parser.add_argument('-ic', '--ignore_column', default=None,
                       help='Comma-separated list of column names to drop from input CSV')
    parser.add_argument('-cb', '--class_balancing', default=True,
                        help='Enable SVM Class Balancing for Training')
    # Model specific arguments
    parser.add_argument('-mt', '--model_type', default='SVM',
                        help='Type of ML model. Currently supported: SVM')
    ## SVM specific
    parser.add_argument('-sk', '--svm_kernel', default='linear',
                       help='SVM kernel type (linear, poly, rbf, sigmoid)')
    ## MLP specific
    ### TBA
    # Misc arguments
    parser.add_argument('-v', '--verbose', type=str, default='False',
                       help='Enable hyperparameter tuning (True/False)')
    
    args = parser.parse_args()
    
    # Convert string arguments to boolean
    tune_hyperparameters = args.verbose.lower() == 'true'
    final_model = args.final.lower() == 'true'
    class_balancing = args.class_balancing.lower() == 'true'
    
    # Parse columns to drop
    if ',' in args.ignore_column:
        ignore_columns = args.ignore_column.split(',')
    elif args.ignore_column==None:
        ignore_columns = None
    else:
        ignore_columns = [args.ignore_column]
    
    try:
        if args.action == 'trainer':
            if not args.model_output:
                raise ValueError("Model output path (-mo) is required for training")
            
            if args.model_type == 'SVM':
                train_svm_model(
                    input_file=args.input,
                    model_path=args.model_output,
                    spare_type=args.type,
                    target_column=args.target_column,
                    kernel=args.svm_kernel,
                    class_balancing=class_balancing,
                    tune_hyperparameters=tune_hyperparameters,
                    final_model=final_model,
                    drop_columns=ignore_columns
                )
            elif args.model_type == 'MLP':
                print("MLP is coming soon!")
            else:
                print(f"{args.model_type} is an unsupported model type.")
            
        elif args.action == 'inference':
            # Check Arguments
            if not args.model:
                raise ValueError("Model path (-m) is required for inference")
            if not args.output:
                raise ValueError("Output path (-o) is required for inference")
            # Run inference
            if args.model_type == 'SVM':
                predict_svm_model(
                    input_file=args.input,
                    model_path=args.model,
                    output_file=args.output,
                    spare_type=args.type,
                    drop_columns=ignore_columns
                )
            elif args.model_type == 'MLP':
                print("MLP is coming soon!")
            else:
                print(f"{args.model_type} is an unsupported model type.")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
