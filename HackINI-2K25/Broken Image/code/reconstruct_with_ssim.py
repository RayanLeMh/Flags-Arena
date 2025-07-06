import numpy as np
import cv2
from PIL import Image
import zipfile
import os
import io
from skimage.metrics import structural_similarity as ssim
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

class SSIMImageReconstructor:
    def __init__(self, zip_path, target_image_path=None):
        """
        Initialize the SSIM Image Reconstructor
        
        Args:
            zip_path: Path to zip file containing tile pieces
            target_image_path: Optional path to target image for comparison
        """
        self.zip_path = zip_path
        self.target_image_path = target_image_path
        self.tiles = []
        self.tile_size = None
        self.grid_dims = None
        
    def load_tiles_from_zip(self):
        """Load all tile pieces from the zip file"""
        print("Loading tiles from zip file...")
        
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            # Get all PNG files and sort them by number
            tile_files = [f for f in zip_ref.namelist() if f.endswith('.png')]
            tile_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            
            for tile_file in tqdm(tile_files[:1023]):  # Load exactly 1023 tiles
                with zip_ref.open(tile_file) as file:
                    img_data = file.read()
                    img = Image.open(io.BytesIO(img_data))
                    img_array = np.array(img)
                    self.tiles.append(img_array)
        
        if self.tiles:
            self.tile_size = self.tiles[0].shape[:2]
            print(f"Loaded {len(self.tiles)} tiles of size {self.tile_size}")
            
            # Calculate grid dimensions for 1023 tiles
            # Find factors close to square root
            sqrt_tiles = int(math.sqrt(1023))
            for i in range(sqrt_tiles, 0, -1):
                if 1023 % i == 0:
                    self.grid_dims = (i, 1023 // i)
                    break
            
            print(f"Grid dimensions: {self.grid_dims}")
    
    def load_tiles_from_folder(self, folder_path):
        """Load all tile pieces from a folder"""
        print(f"Loading tiles from folder: {folder_path}")
        
        # Get all PNG files and sort them by number
        tile_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        tile_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        for tile_file in tqdm(tile_files[:1023]):  # Load exactly 1023 tiles
            img_path = os.path.join(folder_path, tile_file)
            img = Image.open(img_path)
            img_array = np.array(img)
            self.tiles.append(img_array)
        
        if self.tiles:
            self.tile_size = self.tiles[0].shape[:2]
            print(f"Loaded {len(self.tiles)} tiles of size {self.tile_size}")
            
            # Calculate grid dimensions for 1023 tiles
            # Find factors close to square root
            sqrt_tiles = int(math.sqrt(1023))
            for i in range(sqrt_tiles, 0, -1):
                if 1023 % i == 0:
                    self.grid_dims = (i, 1023 // i)
                    break
            
            print(f"Grid dimensions: {self.grid_dims}")
    
    def create_template_grid(self, reference_image=None):
        """
        Create a template grid based on reference image or estimated dimensions
        """
        if reference_image is not None:
            ref_height, ref_width = reference_image.shape[:2]
            tile_h, tile_w = self.tile_size
            
            grid_h = ref_height // tile_h
            grid_w = ref_width // tile_w
            
            self.grid_dims = (grid_h, grid_w)
        
        # Create template positions
        template_positions = []
        grid_h, grid_w = self.grid_dims
        
        for i in range(grid_h):
            for j in range(grid_w):
                if len(template_positions) < 1023:  # Only create 1023 positions
                    template_positions.append((i, j))
        
        return template_positions
    
    def calculate_ssim_matrix(self, reference_tiles):
        """
        Calculate SSIM similarity matrix between tiles and reference positions
        """
        print("Calculating SSIM similarity matrix...")
        
        n_tiles = len(self.tiles)
        n_positions = len(reference_tiles)
        
        ssim_matrix = np.zeros((n_tiles, n_positions))
        
        for i, tile in enumerate(tqdm(self.tiles)):
            for j, ref_tile in enumerate(reference_tiles):
                # Convert to grayscale if needed
                if len(tile.shape) == 3:
                    tile_gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
                else:
                    tile_gray = tile
                    
                if len(ref_tile.shape) == 3:
                    ref_gray = cv2.cvtColor(ref_tile, cv2.COLOR_RGB2GRAY)
                else:
                    ref_gray = ref_tile
                
                # Calculate SSIM
                similarity = ssim(tile_gray, ref_gray, data_range=255)
                ssim_matrix[i, j] = similarity
        
        return ssim_matrix
    
    def solve_assignment_problem(self, ssim_matrix):
        """
        Solve the assignment problem using Hungarian algorithm
        """
        print("Solving assignment problem...")
        
        # Convert similarity to cost (higher similarity = lower cost)
        cost_matrix = 1 - ssim_matrix
        
        # Solve assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        return list(zip(row_indices, col_indices))
    
    def reconstruct_from_reference(self, reference_image):
        """
        Reconstruct image using a reference image for guidance
        """
        print("Reconstructing image from reference...")
        
        # Load reference image
        if isinstance(reference_image, str):
            reference = cv2.imread(reference_image)
            reference = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)
        else:
            reference = reference_image
        
        # Create reference tiles
        ref_tiles = self.extract_reference_tiles(reference)
        
        # Calculate SSIM matrix
        ssim_matrix = self.calculate_ssim_matrix(ref_tiles)
        
        # Solve assignment
        assignments = self.solve_assignment_problem(ssim_matrix)
        
        # Reconstruct image
        reconstructed = self.assemble_image(assignments, ref_tiles)
        
        return reconstructed, assignments
    
    def extract_reference_tiles(self, reference_image):
        """
        Extract tiles from reference image for comparison
        """
        ref_height, ref_width = reference_image.shape[:2]
        tile_h, tile_w = self.tile_size
        
        ref_tiles = []
        positions = self.create_template_grid(reference_image)
        
        for pos_i, pos_j in positions:
            start_i = pos_i * tile_h
            end_i = start_i + tile_h
            start_j = pos_j * tile_w
            end_j = start_j + tile_w
            
            if end_i <= ref_height and end_j <= ref_width:
                tile = reference_image[start_i:end_i, start_j:end_j]
                ref_tiles.append(tile)
        
        return ref_tiles
    
    def assemble_image(self, assignments, reference_tiles):
        """
        Assemble the final image based on assignments
        """
        print("Assembling final image...")
        
        grid_h, grid_w = self.grid_dims
        tile_h, tile_w = self.tile_size
        
        # Create output image
        if len(self.tiles[0].shape) == 3:
            channels = self.tiles[0].shape[2]
            reconstructed = np.zeros((grid_h * tile_h, grid_w * tile_w, channels), dtype=np.uint8)
        else:
            reconstructed = np.zeros((grid_h * tile_h, grid_w * tile_w), dtype=np.uint8)
        
        # Place tiles according to assignments
        for tile_idx, ref_idx in assignments:
            # Calculate grid position from reference index
            grid_i = ref_idx // grid_w
            grid_j = ref_idx % grid_w
            
            # Calculate pixel position
            start_i = grid_i * tile_h
            end_i = start_i + tile_h
            start_j = grid_j * tile_w
            end_j = start_j + tile_w
            
            # Place tile
            if end_i <= reconstructed.shape[0] and end_j <= reconstructed.shape[1]:
                reconstructed[start_i:end_i, start_j:end_j] = self.tiles[tile_idx]
        
        return reconstructed
    
    def reconstruct_blind(self):
        """
        Reconstruct image without reference (blind reconstruction)
        Uses edge matching and correlation
        """
        print("Performing blind reconstruction...")
        
        # This is a simplified approach - in practice, blind reconstruction is very complex
        # We'll use a greedy approach based on edge similarity
        
        grid_h, grid_w = self.grid_dims
        tile_h, tile_w = self.tile_size
        
        # Create output image
        if len(self.tiles[0].shape) == 3:
            channels = self.tiles[0].shape[2]
            reconstructed = np.zeros((grid_h * tile_h, grid_w * tile_w, channels), dtype=np.uint8)
        else:
            reconstructed = np.zeros((grid_h * tile_h, grid_w * tile_w), dtype=np.uint8)
        
        # Simple placement (this would need more sophisticated edge matching)
        used_tiles = set()
        
        for i in range(grid_h):
            for j in range(grid_w):
                pos_idx = i * grid_w + j
                if pos_idx < len(self.tiles) and pos_idx not in used_tiles:
                    start_i = i * tile_h
                    end_i = start_i + tile_h
                    start_j = j * tile_w
                    end_j = start_j + tile_w
                    
                    reconstructed[start_i:end_i, start_j:end_j] = self.tiles[pos_idx]
                    used_tiles.add(pos_idx)
        
        return reconstructed
    
    def visualize_results(self, original, reconstructed, save_path=None):
        """
        Visualize original and reconstructed images
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        axes[0].imshow(original)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(reconstructed)
        axes[1].set_title('Reconstructed Image')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

# Usage example
def main():
    # Initialize reconstructor with your actual file paths
    reconstructor = SSIMImageReconstructor('tiles.zip')
    
    # Try to load tiles from zip first, then from folder
    if os.path.exists('tiles.zip'):
        reconstructor.load_tiles_from_zip()
    elif os.path.exists('tiles'):
        reconstructor.zip_path = 'tiles'  # Update path for folder
        reconstructor.load_tiles_from_folder('tiles')
    else:
        print("Error: Could not find 'tiles.zip' or 'tiles' folder")
        return
    
    # Option 1: Reconstruct with reference image
    if os.path.exists('original.jpg'):
        reconstructed, assignments = reconstructor.reconstruct_from_reference('original.jpg')
        
        # Load original for comparison
        original = cv2.imread('original.jpg')
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Visualize results
        reconstructor.visualize_results(original, reconstructed, 'reconstruction_result.png')
        
        # Calculate final SSIM score
        final_ssim = ssim(
            cv2.cvtColor(original, cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(reconstructed, cv2.COLOR_RGB2GRAY),
            data_range=255
        )
        print(f"Final SSIM score: {final_ssim:.4f}")
    
    # Option 2: Blind reconstruction
    else:
        print("No reference image found. Performing blind reconstruction...")
        reconstructed = reconstructor.reconstruct_blind()
        
        # Save result
        result_image = Image.fromarray(reconstructed)
        result_image.save('blind_reconstruction_result.png')
        print("Blind reconstruction saved as 'blind_reconstruction_result.png'")

if __name__ == "__main__":
    main()