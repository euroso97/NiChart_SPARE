#!/usr/bin/env python3
"""
NiChart_SPARE - Main entry point for SPARE scores calculation

This script provides command-line interface for training and inference of SPARE models.
Supported SPARE types: BA (Brain Age), AD (Alzheimer's), HT (Hypertension), HL (Hyperlipidemia), 
T2B (Diabetes), SM (Smoking), OB (Obesity)
"""

import argparse
import sys
import pandas as pd
from .pipelines import spare_ad, spare_ba, spare_ht

def load_csv_data(file_path, drop_columns=None):
    """Load CSV data and return dataframe, optionally dropping specified columns"""
    df = pd.read_csv(file_path)
    print(f"Loaded data: {len(df)} samples, {len(df.columns)} columns")
    if drop_columns:
        for col in drop_columns:
            if col in df.columns:
                df = df.drop(columns=[col])
                print(f"Dropped column: {col}")
            else:
                print(f"Warning: Column to drop not found: {col}")
    return df

def get_pipeline_module(spare_type):
    """Get the appropriate pipeline module based on SPARE type"""
    spare_type = spare_type.upper()
    
    pipeline_map = {
        'AD': spare_ad,
        'BA': spare_ba,
        'HT': spare_ht,
    }
    
    if spare_type not in pipeline_map:
        raise ValueError(f"Unsupported SPARE type: {spare_type}")
    
    return pipeline_map[spare_type]

def train_model(input_file, 
                model_path, 
                spare_type, 
                target_column, 
                kernel, 
                tune_hyperparameters, 
                final_model, 
                drop_columns=None):
    """Train model using the pipeline functions"""
    # Load data
    print("Loading training data...")
    df = load_csv_data(input_file, drop_columns=drop_columns)
    
    # Validate target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found. Available columns: {list(df.columns)}")
    
    # Get pipeline module
    pipeline_module = get_pipeline_module(spare_type)
    
    if spare_type=='BA':
        pass
        # Regression model (BA - Brain Age)
        # Standard training
        # print("Training model with default parameters...")
        # model, scaler, info = pipeline_module.train_svr_model(
        #     dataframe=df,
        #     target_column=target_column,
        #     kernel=kernel,
        #     tune_hyperparameters=False
        # )
        
        # # Save model
        # pipeline_module.save_model(model, scaler, info, model_path)
        # print(f"Model saved to: {model_path}")
    
    elif spare_type=='AD':        
        # Standard training
        print("Training model with default parameters...")
        model, feature_encoder, label_encoder, scaler = pipeline_module.train_svc_model(
            dataframe=df,
            target_column=target_column,
            kernel=kernel,
            tune_hyperparameters=False,
            train_whole_set=True
        )
        # Save model
        from .util import save_model
        save_model(model = model, 
                   scaler = scaler, 
                   training_info = {}, 
                   filepath = model_path)
        print(f"Model saved to: {model_path}")
    else:
        print(f"{spare_type} is not supported.")
        sys.exit(1)

def is_regression_model(spare_type):
    """Check if the SPARE type uses regression (continuous target)"""
    regression_types = ['BA']  # Brain Age is continuous
    return spare_type.upper() in regression_types

def predict_model(input_file, 
                  model_path, 
                  output_file, 
                  spare_type, 
                  drop_columns=None):
    """Make predictions using trained model"""
    
    # Load data
    print("Loading prediction data...")
    df = load_csv_data(input_file, drop_columns=drop_columns)
    
    # Get pipeline module
    pipeline_module = get_pipeline_module(spare_type)
    
    # Load model
    print("Loading trained model...")
    
    if is_regression_model(spare_type):
        # Regression model
        model, scaler, info = pipeline_module.load_model(model_path)
        predictions = pipeline_module.predict_svr(model, scaler, df)
    else:
        # Classification model
        model, scaler, encoder, info = pipeline_module.load_model(model_path)
        predictions = pipeline_module.predict_svc(model, scaler, df, encoder)
    
    # Create output dataframe
    output_df = df.copy()
    output_df['predicted_target'] = predictions
    
    # Save predictions
    output_df.to_csv(output_file, index=False)
    print(f"Predictions saved to: {output_file}")


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
    
    # Add arguments
    parser.add_argument('-a', '--action', required=True, choices=['trainer', 'inference'],
                       help='Action to perform: trainer or inference')
    parser.add_argument('-t', '--type', required=True, 
                       help='SPARE type: BA (Brain Age), AD (Alzheimer\'s), HT (Hypertension)')
    parser.add_argument('-i', '--input', required=True,
                       help='Input CSV file path')
    parser.add_argument('-mo', '--model_output', 
                       help='Output model file path (for training)')
    parser.add_argument('-m', '--model', 
                       help='Input model file path (for inference)')
    parser.add_argument('-v', '--verbose', type=str, default='False',
                       help='Enable hyperparameter tuning (True/False)')
    parser.add_argument('-f', '--final', type=str, default='False',
                       help='Train final model on entire dataset (True/False)')
    parser.add_argument('-o', '--output', 
                       help='Output CSV file path (for inference)')
    parser.add_argument('-k', '--kernel', default='linear',
                       help='SVM kernel type (linear, poly, rbf, sigmoid)')
    parser.add_argument('-kv', '--key_variable',
                       help='Name of column indicating unique data points in the input CSV')
    parser.add_argument('-tc', '--target_column', default='target',
                       help='Name of target column in CSV')
    parser.add_argument('-iv', '--input_drop', default=None,
                       help='Comma-separated list of column names to drop from input CSV')
    
    args = parser.parse_args()
    
    # Convert string arguments to boolean
    tune_hyperparameters = args.verbose.lower() == 'true'
    final_model = args.final.lower() == 'true'
    # Parse columns to drop
    drop_columns = [col.strip() for col in args.input_drop.split(',')] if args.input_drop else None
    
    try:
        if args.action == 'trainer':
            if not args.model_output:
                raise ValueError("Model output path (-mo) is required for training")
            
            train_model(
                input_file=args.input,
                model_path=args.model_output,
                spare_type=args.type,
                target_column=args.target_column,
                kernel=args.kernel,
                tune_hyperparameters=tune_hyperparameters,
                final_model=final_model,
                drop_columns=drop_columns
            )
            
        elif args.action == 'inference':
            if not args.model:
                raise ValueError("Model path (-m) is required for inference")
            if not args.output:
                raise ValueError("Output path (-o) is required for inference")
            
            predict_model(
                input_file=args.input,
                model_path=args.model,
                output_file=args.output,
                spare_type=args.type,
                drop_columns=drop_columns
            )
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
