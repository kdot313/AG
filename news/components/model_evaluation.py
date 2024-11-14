import os
import sys
import numpy as np
import torch
import pandas as pd
from news.logger import logging
from news.exception import CustomException
from torch.utils.data import DataLoader
from datasets import Dataset
from typing import Dict
from news.constants import *
from news.configuration.s3_operations import S3Operation
from sklearn.metrics import accuracy_score, f1_score
from safetensors import safe_open
from news.entity.config_entity import ModelEvaluationConfig, ModelTrainerConfig
from news.entity.artifact_entity import ModelEvaluationArtifacts, ModelTrainerArtifacts, DataTransformationArtifacts
from news.ml.model import RobertaModel

class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 model_trainer_artifacts: ModelTrainerArtifacts,
                 data_transformation_artifacts: DataTransformationArtifacts,
                 model_trainer_config: ModelTrainerConfig):
        
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config
        self.robertamodel = RobertaModel(model_trainer_config)
        self.tokenizer = self.robertamodel.tokenizer
        self.awscloud = S3Operation()


    def get_best_model_from_aws(self) -> str:
        """
        :return: Fetch best model from AWS S3 storage and store inside best model directory path
        """
        try:
            logging.info("Entered the get_best_model_from_aws method of Model Evaluation class")

            os.makedirs(self.model_evaluation_config.BEST_MODEL_DIR_PATH, exist_ok=True)

            best_model_path = self.model_evaluation_config.BEST_MODEL_DIR_PATH
            
            self.awscloud.download_folder(local_dir = best_model_path,
                                          folder_key = BEST_MODEL_DIR,
                                          bucket_name = self.model_evaluation_config.BUCKET_NAME)

            logging.info("Exited the get_best_model_from_aws method of Model Evaluation class")
            return best_model_path

        except Exception as e:
            logging.warning("Best model not found on AWS, will proceed with locally trained model.")
            return None  # Return None to handle AWS model absence gracefully
        

    
    def evaluate_model(self, model_path: str, test_data_loader: DataLoader) -> float:
        """
        Evaluates a given model on the test dataset.
        Returns the accuracy of the model.
        """
        try:
            model = self.robertamodel.model
            state_dict = {}

            # Open SafeTensors file and load weights into state_dict
            with safe_open(model_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            
            model.load_state_dict(state_dict)
            model.eval()

            predictions, true_labels = [], []

            with torch.no_grad():
                for batch in test_data_loader:
                    inputs = batch['input_ids']
                    masks = batch['attention_mask']
                    labels = batch['label']
                    
                    outputs = model(inputs, masks)
                    _, preds = torch.max(outputs.logits, dim=1)
                    
                    predictions.extend(preds.tolist())
                    true_labels.extend(labels.tolist())

            accuracy = accuracy_score(true_labels, predictions)
            f1 = f1_score(true_labels, predictions, average='weighted')

            logging.info(f"Model Evaluation - Accuracy: {accuracy}, F1 Score: {f1}")
            return accuracy, f1

        except Exception as e:
            raise CustomException(e, sys) from e
        
    def tokenize_data(self, examples: Dict) -> Dict:
        """Tokenizes data for input to the model."""
        return self.tokenizer(examples[self.model_trainer_config.TEXT], 
                              padding=self.model_trainer_config.PADDING, 
                              truncation=True, 
                              max_length=self.model_trainer_config.MAX_SEQ_LENGTH)
    
    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        """
        Compares the AWS best model with the local model, and retains the better-performing model.
        """
        try:
            logging.info("Starting model evaluation process.")

            # Load test data
            test_df = pd.read_csv(self.data_transformation_artifacts.transformed_test_data_path)
            test_dataset = Dataset.from_pandas(test_df)
            test_dataset = test_dataset.map(self.tokenize_data, batched=True)
            test_loader = DataLoader(test_dataset, batch_size=self.model_evaluation_config.EVAL_BATCH_SIZE)

            # Evaluate AWS best model if available
            best_model_path = self.get_best_model_from_aws()
            print("#######################################*************")
            print(best_model_path)
            print("#######################################*************")
            if best_model_path and os.path.exists(os.path.join(best_model_path, 'model.safetensors')):
                aws_accuracy, aws_f1 = self.evaluate_model(os.path.join(best_model_path, 'model.safetensors'), test_loader)
            else:
                aws_accuracy, aws_f1 = 0, 0  # Default values if AWS model isn't available

            # Evaluate locally trained model
            local_accuracy, local_f1 = self.evaluate_model(os.path.join(self.model_trainer_artifacts.trained_model_path, 'model.safetensors'), test_loader)

            # Determine the better model
            if local_accuracy >= aws_accuracy and local_f1 >= aws_f1:
                best_model_path = self.model_trainer_artifacts.trained_model_path
                is_model_accepted = True
                logging.info("Local model chosen as the best model.")
            else:
                is_model_accepted = False
                logging.info("AWS model chosen as the best model.")

            # Return Model Evaluation Artifacts
            model_evaluation_artifacts = ModelEvaluationArtifacts(
                trained_model_accuracy=max(local_accuracy, aws_accuracy),
                is_model_accepted=is_model_accepted,
                best_model_path=best_model_path
            )
            logging.info("Model evaluation completed successfully.")
            return model_evaluation_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e