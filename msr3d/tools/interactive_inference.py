import torch
import yaml
from pathlib import Path
import sys
import os

# Add the parent directory to the path so we can import from msr3d
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.build import build_model
from model.msr3d.msr3d import MSR3D
from common.type_utils import cfg2dict
from common.io_utils import load_point_cloud
from data.data_utils import load_scene_data

class InteractiveMSR3D:
    def __init__(self, experiment_path):
        """
        Initialize the MSR3D model for interactive inference
        
        Args:
            experiment_path: Path to experiment folder containing config.yaml and model weights
        """
        self.experiment_path = Path(experiment_path)
        self.config_path = self.experiment_path / "config.yaml"
        self.model_path = self.experiment_path / "best.pth" / "pytorch_model.bin"
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Convert to proper config format expected by model
        from common.type_utils import Dict2Config
        self.config = Dict2Config(self.config)
        
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = build_model(self.config)
        
        # Load model weights - handle .bin file directly as state dict
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def prepare_input(self, scene_id, question):
        """
        Prepare the input data dictionary for the model
        
        Args:
            scene_id: ID of the scene to load
            question: Question to ask about the scene
            
        Returns:
            data_dict: Dictionary containing model inputs
        """
        # Load scene data
        scene_data = load_scene_data(scene_id)
        point_cloud = load_point_cloud(scene_id)
        
        data_dict = {
            'scan_id': scene_id,
            'point_cloud': torch.from_numpy(point_cloud).float().to(self.device),
            'situation_text': scene_data['situation_text'],
            'question': question,
            'img_fts': scene_data.get('img_fts', None)
        }
        
        # Add any additional required fields from scene_data
        if 'obj_fts' in scene_data:
            data_dict['obj_fts'] = torch.from_numpy(scene_data['obj_fts']).float().to(self.device)
        if 'obj_masks' in scene_data:
            data_dict['obj_masks'] = torch.from_numpy(scene_data['obj_masks']).to(self.device)
        
        return data_dict

    @torch.no_grad()
    def ask_question(self, scene_id, question):
        """
        Ask a question about a scene
        
        Args:
            scene_id: ID of the scene to query
            question: Question to ask about the scene
            
        Returns:
            answer: Model's answer to the question
        """
        data_dict = self.prepare_input(scene_id, question)
        
        # Generate answer using model
        outputs = self.model.generate(
            data_dict,
            use_nucleus_sampling=True,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=3.0,
            length_penalty=1,
            temperature=0.5
        )
        
        return outputs['answer']

def main():
    # Example usage
    experiment_path = "../MSR3D_BLIP_PNPP_ViC_LORA_TUNED"  # Path to directory containing pytorch_model.bin
    inferencer = InteractiveMSR3D(experiment_path)
    
    print("MSR3D Interactive Question Answering")
    print("====================================")
    print(f"Model loaded from: {experiment_path}")
    print("\nInstructions:")
    print("- Enter 'quit' to exit")
    print("- Enter 'change scene' to load a different scene")
    
    current_scene = None
    while True:
        if current_scene is None:
            scene_id = input("\nEnter scene ID: ")
            if scene_id.lower() == 'quit':
                break
            current_scene = scene_id
        
        question = input("\nEnter your question (or 'change scene'/'quit'): ")
        if question.lower() == 'quit':
            break
        elif question.lower() == 'change scene':
            current_scene = None
            continue
            
        try:
            answer = inferencer.ask_question(current_scene, question)
            print(f"\nAnswer: {answer}")
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please try again with a different question or scene ID")

if __name__ == "__main__":
    main()


#sshfs pdemetriou@prometheus.cyens.org.cy:/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/MSR3D_BLIP_PNPP_ViC_LORA_TUNED /home/panagiotis/msqa/Msqa_Thesis_2025/msr3d/MSR3D_BLIP_PNPP_ViC_LORA_TUNED/