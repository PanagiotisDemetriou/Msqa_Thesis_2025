import torch
import yaml
from pathlib import Path

class MSR3DInference:
    def __init__(self, model_path, config_path):
        """
        Initialize the MSR3D model for inference
        
        Args:
            model_path: Path to model.bin file
            config_path: Path to YAML config file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model checkpoint
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model architecture (you'll need to import your model class)
        # from msr3d.models import MSR3DModel  # Adjust import based on repo structure
        # self.model = MSR3DModel(self.config)
        
        # Load model weights
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.load_state_dict(checkpoint)  # If checkpoint is just the state dict
        self.model.to(self.device)
        self.model.eval()
        
    def answer_question(self, point_cloud, situation_text, question, 
                       situation_images=None, location=None, orientation=None):
        """
        Answer a question about the 3D scene
        
        Args:
            point_cloud: 3D point cloud data (numpy array or tensor)
            situation_text: Text describing the situation
            question: The question to answer
            situation_images: Optional images related to the situation
            location: Optional location in the scene
            orientation: Optional orientation information
            
        Returns:
            answer: Model's answer to the question
        """
        with torch.no_grad():
            # Prepare inputs (adjust based on your model's input format)
            inputs = {
                'point_cloud': torch.tensor(point_cloud).to(self.device),
                'situation_text': situation_text,
                'question': question,
            }
            
            if situation_images is not None:
                inputs['images'] = torch.tensor(situation_images).to(self.device)
            if location is not None:
                inputs['location'] = torch.tensor(location).to(self.device)
            if orientation is not None:
                inputs['orientation'] = torch.tensor(orientation).to(self.device)
            
            # Run inference
            outputs = self.model(**inputs)
            
            # Process output (adjust based on your model's output format)
            answer = self.process_output(outputs)
            
        return answer
    
    def process_output(self, outputs):
        """Process model outputs to extract the answer"""
        # This depends on your model's output format
        # It might be text generation, classification, or other formats
        if isinstance(outputs, dict) and 'answer' in outputs:
            return outputs['answer']
        return outputs

# Usage example
if __name__ == "__main__":
    # Initialize the inference model
    inferencer = MSR3DInference(
        #sshfs pdemetriou@prometheus.cyens.org.cy:/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/MSR3D_BLIP_PNPP_ViC_LORA_TUNED /home/panagiotis/msqa/Msqa_Thesis_2025/msr3d/MSR3D_BLIP_PNPP_ViC_LORA_TUNED/
        model_path="../MSR3D_BLIP_PNPP_ViC_LORA_TUNED/best.pth/pytorch_model.bin",  # Path to your trained model
        config_path="../MSR3D_BLIP_PNPP_ViC_LORA_TUNED/config.yaml"  # Make sure you have the config file
    )
    
    # Load your 3D scene data
    # point_cloud = load_point_cloud("path/to/scene.ply")  # Your data loading function
    
    # Ask a question
    situation = "I am standing in a living room facing the sofa."
    question = "What is on my left?"
    
    # answer = inferencer.answer_question(
    #     point_cloud=point_cloud,
    #     situation_text=situation,
    #     question=question
    # )
    
    # print(f"Question: {question}")
    # print(f"Answer: {answer}")
