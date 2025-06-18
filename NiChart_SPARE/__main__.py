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

def load_csv_data(file_path):
    """Load CSV data and return dataframe"""
    df = pd.read_csv(file_path)
    print(f"Loaded data: {len(df)} samples, {len(df.columns)} columns")
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

def is_regression_model(spare_type):
    """Check if the SPARE type uses regression (continuous target)"""
    regression_types = ['BA']  # Brain Age is continuous
    return spare_type.upper() in regression_types

def train_model(input_file, model_path, spare_type, target_column, kernel, tune_hyperparameters, final_model):
    """Train model using the pipeline functions"""
    
    # Load data
    print("Loading training data...")
    df = load_csv_data(input_file)
    
    # Validate target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found. Available columns: {list(df.columns)}")
    
    # Get pipeline module
    pipeline_module = get_pipeline_module(spare_type)
    is_regression = is_regression_model(spare_type)
    
    if is_regression:
        # Regression model (BA - Brain Age)
        if final_model and tune_hyperparameters:
            # Complete workflow: tuning + final model
            print("Step 1: Hyperparameter tuning...")
            model_tuned, scaler_tuned, info_tuned = pipeline_module.train_svr_model(
                dataframe=df,
                target_column=target_column,
                kernel=kernel,
                tune_hyperparameters=True
            )
            
            print(f"Best parameters found: {info_tuned['best_params']}")
            print(f"CV R² score: {info_tuned['cv_score']:.3f}")
            
            print("Step 2: Training final model on entire dataset...")
            final_model, final_scaler, final_info = pipeline_module.train_final_model(
                dataframe=df,
                target_column=target_column,
                best_params=info_tuned['best_params'],
                kernel=kernel
            )
            
            # Save final model
            pipeline_module.save_model(final_model, final_scaler, final_info, model_path)
            print(f"Final model saved to: {model_path}")
            
        elif tune_hyperparameters:
            # Only hyperparameter tuning
            print("Performing hyperparameter tuning...")
            model, scaler, info = pipeline_module.train_svr_model(
                dataframe=df,
                target_column=target_column,
                kernel=kernel,
                tune_hyperparameters=True
            )
            
            # Save tuned model
            pipeline_module.save_model(model, scaler, info, model_path)
            print(f"Tuned model saved to: {model_path}")
            
        else:
            # Standard training
            print("Training model with default parameters...")
            model, scaler, info = pipeline_module.train_svr_model(
                dataframe=df,
                target_column=target_column,
                kernel=kernel,
                tune_hyperparameters=False
            )
            
            # Save model
            pipeline_module.save_model(model, scaler, info, model_path)
            print(f"Model saved to: {model_path}")
    
    else:
        # Classification model (AD, HT)
        if final_model and tune_hyperparameters:
            # Complete workflow: tuning + final model
            print("Step 1: Hyperparameter tuning...")
            model_tuned, scaler_tuned, encoder_tuned, info_tuned = pipeline_module.train_svc_model(
                dataframe=df,
                target_column=target_column,
                kernel=kernel,
                tune_hyperparameters=True
            )
            
            print(f"Best parameters found: {info_tuned['best_params']}")
            print(f"CV accuracy: {info_tuned['cv_score']:.3f}")
            
            print("Step 2: Training final model on entire dataset...")
            final_model, final_scaler, final_encoder, final_info = pipeline_module.train_final_model(
                dataframe=df,
                target_column=target_column,
                best_params=info_tuned['best_params'],
                kernel=kernel
            )
            
            # Save final model
            pipeline_module.save_model(final_model, final_scaler, final_encoder, final_info, model_path)
            print(f"Final model saved to: {model_path}")
            
        elif tune_hyperparameters:
            # Only hyperparameter tuning
            print("Performing hyperparameter tuning...")
            model, scaler, encoder, info = pipeline_module.train_svc_model(
                dataframe=df,
                target_column=target_column,
                kernel=kernel,
                tune_hyperparameters=True
            )
            
            # Save tuned model
            pipeline_module.save_model(model, scaler, encoder, info, model_path)
            print(f"Tuned model saved to: {model_path}")
            
        else:
            # Standard training
            print("Training model with default parameters...")
            model, scaler, encoder, info = pipeline_module.train_svc_model(
                dataframe=df,
                target_column=target_column,
                kernel=kernel,
                tune_hyperparameters=False
            )
            
            # Save model
            pipeline_module.save_model(model, scaler, encoder, info, model_path)
            print(f"Model saved to: {model_path}")

def predict_model(input_file, model_path, output_file, spare_type):
    """Make predictions using trained model"""
    
    # Load data
    print("Loading prediction data...")
    df = load_csv_data(input_file)
    
    # Get pipeline module
    pipeline_module = get_pipeline_module(spare_type)
    is_regression = is_regression_model(spare_type)
    
    # Load model
    print("Loading trained model...")
    if is_regression:
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

def main():
    """Main entry point for NiChart_SPARE"""
    parser = argparse.ArgumentParser(
        description="NiChart_SPARE - SPARE scores calculation from Brain ROI Volumes",
        epilog="""
Examples:
  # Training with default parameters
  NiChart_SPARE -a trainer -t AD -i input.csv -mo model.pkl -tc target_column
  
  # Training with hyperparameter tuning
  NiChart_SPARE -a trainer -t BA -i input.csv -mo model.pkl -tc target_column --tune
  
  # Training with hyperparameter tuning + final model
  NiChart_SPARE -a trainer -t HT -i input.csv -mo model.pkl -tc target_column --tune --final
  
  # Inference
  NiChart_SPARE -a inference -t AD -i test.csv -mo model.pkl -o predictions.csv
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
    
    # Training-specific arguments
    parser.add_argument('-tc', '--target_column',
                       help='Target column name (required for training)')
    
    parser.add_argument('--kernel',
                       default='rbf',
                       choices=['linear', 'poly', 'rbf', 'sigmoid'],
                       help='SVC kernel type (default: rbf)')
    
    parser.add_argument('--tune',
                       action='store_true',
                       help='Perform hyperparameter tuning with 5-fold CV')
    
    parser.add_argument('--final',
                       action='store_true',
                       help='Train final model on entire dataset after tuning')
    
    # Inference-specific arguments
    parser.add_argument('-o', '--output',
                       help='Output file path (required for inference)')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Execute the appropriate action
        if args.action == 'trainer':
            if not args.target_column:
                raise ValueError("Target column (-tc) is required for training")
            
            # Train model
            train_model(
                input_file=args.input,
                model_path=args.model,
                spare_type=args.type,
                target_column=args.target_column,
                kernel=args.kernel,
                tune_hyperparameters=args.tune,
                final_model=args.final
            )
                
        elif args.action == 'inference':
            if not args.output:
                raise ValueError("Output file path (-o) is required for inference")
            
            # Make predictions
            predict_model(
                input_file=args.input,
                model_path=args.model,
                output_file=args.output,
                spare_type=args.type
            )
        
        print("Operation completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
