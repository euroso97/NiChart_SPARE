# import sys

# from .util import load_csv_data, get_pipeline_module, is_regression_model
# from .svm import save_svm_model, load_svm_model

# def train_svm_model(input_file, 
# 					model_path, 
# 					spare_type, 
# 					target_column, 
# 					kernel, 
# 					tune_hyperparameters, 
# 					final_model, 
# 					drop_columns=None):
#     """Train model using the pipeline functions"""
#     # Load data
#     print("Loading training data...")
#     df = load_csv_data(input_file, drop_columns=drop_columns)
    
#     # Validate target column exists
#     if target_column not in df.columns:
#         raise ValueError(f"Target column '{target_column}' not found. Available columns: {list(df.columns)}")
    
#     # Get pipeline module
#     pipeline_module = get_pipeline_module(spare_type)
    
#     if spare_type.isin(['BA']):
#         model, scaler = None
#         # Regression model (BA - Brain Age)
#         if kernel=='linear':
#             print("Training model with LinearSVR...")
#             model, scaler = pipeline_module.train_linearsvr_model(
#                 dataframe=df,
#                 target_column=target_column,
#                 kernel=kernel,
#                 tune_hyperparameters=False,
#                 train_whole_set=True
#             )
#         else:
#             # Standard training
#             print("Training model with default SVR...")
#             model, scaler = pipeline_module.train_svr_model(
#                 dataframe=df,
#                 target_column=target_column,
#                 kernel=kernel,
#                 tune_hyperparameters=False,
#                 train_whole_set=True
#             )
        
#         if model != None & scaler != None & model_path != None:
#             # Save model
#             save_svm_model(model, scaler, {}, model_path)
#             print(f"Model saved to: {model_path}")
    
#     elif spare_type.isin(['AD']):
#         if kernel.lower()=='linear':
#             print("Training model with LinearSVR...")
#             model, feature_encoder, label_encoder, scaler = pipeline_module.train_linearsvc_model()
#         elif kernel.lower().isin(['poly', 'rbf', 'sigmoid']):
#             model, feature_encoder, label_encoder, scaler = pipeline_module.train_svc_model(dataframe=df,
#                                                                                             target_column=target_column,
#                                                                                             kernel=kernel,
#                                                                                             tune_hyperparameters=False,
#                                                                                             get_cv_scores=True,
#                                                                                             train_whole_set=True)
#             pipeline_module.save_svm_model(model = model, 
#                                            scaler = scaler, 
#                                            training_info = {}, 
#                                            filepath = model_path)
#             print(f"Model saved to: {model_path}")
#         else:
#             print(f"Unsupported SVM kernel entry. Please select among: linear, poly, rbf, sigmoid.")
    
#     else:
#         print(f"{spare_type} is not supported.")
#         sys.exit(1)


# def predict_svm_model(input_file, 
#                       model_path, 
#                       output_file, 
#                       spare_type, 
#                       drop_columns=None):
#     """Make predictions using trained model"""
    
#     # Load data
#     print("Loading prediction data...")
#     df = load_csv_data(input_file, drop_columns=drop_columns)
    
#     # Get pipeline module
#     pipeline_module = get_pipeline_module(spare_type)
    
#     # Load model
#     print("Loading trained model...")
    
#     if is_regression_model(spare_type):
#         # Regression model
#         model, scaler, info = pipeline_module.load_model(model_path) # TBF
#         predictions = pipeline_module.predict_svr(model, scaler, df)
#     else:
#         # Classification model
#         model, scaler, encoder, info = pipeline_module.load_model(model_path) # TBF
#         predictions = pipeline_module.predict_svc(model, scaler, df, encoder)
    
#     # Create output dataframe
#     output_df = df.copy()
#     output_df['predicted_target'] = predictions
    
#     # Save predictions
#     output_df.to_csv(output_file, index=False)
#     print(f"Predictions saved to: {output_file}")