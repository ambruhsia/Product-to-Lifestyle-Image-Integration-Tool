import os
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
from rembg import remove
import random

class ProductLifestyleIntegrator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {self.device}")
        
        # Load pre-generated backgrounds to avoid complexity
        self.backgrounds = {
            "living_room": ["backgrounds/living_room1.jpg", "backgrounds/living_room2.jpg"],
            "bedroom": ["backgrounds/bedroom1.jpg", "backgrounds/bedroom2.jpg"],
            "dining_room": ["backgrounds/dining_room1.jpg", "backgrounds/dining_room2.jpg"],
            "office": ["backgrounds/office1.jpg", "backgrounds/office2.jpg"]
        }
        
        # Create placement zones for different room types
        self.placement_zones = {
            "living_room": [
                {"name": "coffee_table", "box": [200, 300, 400, 350]},
                {"name": "shelf", "box": [100, 150, 300, 200]},
                {"name": "side_table", "box": [500, 280, 600, 340]}
            ],
            "bedroom": [
                {"name": "bedside_table", "box": [150, 250, 300, 320]},
                {"name": "dresser", "box": [400, 200, 550, 280]},
                {"name": "bed", "box": [250, 350, 450, 450]}
            ],
            "dining_room": [
                {"name": "dining_table", "box": [250, 280, 450, 350]},
                {"name": "sideboard", "box": [500, 250, 650, 320]},
                {"name": "shelf", "box": [150, 150, 250, 220]}
            ],
            "office": [
                {"name": "desk", "box": [200, 280, 400, 350]},
                {"name": "shelf", "box": [500, 200, 600, 280]},
                {"name": "side_table", "box": [100, 300, 180, 350]}
            ]
        }
        
        # Map product types to appropriate rooms
        self.product_to_room = {
            "vase": "living_room",
            "lamp": "bedroom",
            "cushion": "living_room",
            "pillow": "bedroom",
            "chair": "dining_room",
            "table": "dining_room",
            "mirror": "bedroom",
            "clock": "office",
            "rug": "living_room",
            "artwork": "living_room",
            "curtain": "living_room",
            "bookshelf": "office",
            "sofa": "living_room",
            "bed": "bedroom",
            "desk": "office"
        }
            
    def remove_background(self, product_image_path):
        """Remove background from product image using rembg"""
        print(f"Removing background from {product_image_path}")
        input_image = Image.open(product_image_path)
        output_image = remove(input_image)
        return output_image
    
    def load_background(self, room_type):
        """Load a background image based on room type"""
        if room_type in self.backgrounds and self.backgrounds[room_type]:
            # Select a random background from the options for this room type
            bg_path = random.choice(self.backgrounds[room_type])
            
            # Check if the file exists, if not use a placeholder
            if os.path.exists(bg_path):
                bg_img = cv2.imread(bg_path)
                bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            else:
                # Create a placeholder background if file doesn't exist
                bg_img = np.ones((768, 1024, 3), dtype=np.uint8) * 240  # Light gray
                
                # Add some basic elements to simulate a room
                if room_type == "living_room":
                    # Draw a sofa
                    cv2.rectangle(bg_img, (200, 400), (800, 600), (180, 180, 220), -1)
                    # Draw a coffee table
                    cv2.rectangle(bg_img, (400, 620), (600, 700), (120, 80, 60), -1)
                elif room_type == "bedroom":
                    # Draw a bed
                    cv2.rectangle(bg_img, (200, 300), (800, 600), (200, 200, 240), -1)
                    # Draw a bedside table
                    cv2.rectangle(bg_img, (150, 350), (180, 400), (120, 80, 60), -1)
                elif room_type == "dining_room":
                    # Draw a dining table
                    cv2.rectangle(bg_img, (300, 350), (700, 550), (120, 80, 60), -1)
                elif room_type == "office":
                    # Draw a desk
                    cv2.rectangle(bg_img, (250, 400), (750, 550), (120, 80, 60), -1)
                
                # Add a simple wall and floor division
                cv2.line(bg_img, (0, 350), (1024, 350), (200, 200, 200), 2)
        else:
            # Create a default background if the room type is not found
            bg_img = np.ones((768, 1024, 3), dtype=np.uint8) * 240  # Light gray
            
        return bg_img, room_type
    
    def get_placement_zones(self, room_type, background_img):
        """Get suitable placement zones for the given room type"""
        if room_type in self.placement_zones:
            placement_zones = self.placement_zones[room_type]
        else:
            # Create default placement zones based on image size
            height, width = background_img.shape[:2]
            placement_zones = [
                {"name": "center", "box": [width//4, height//3, 3*width//4, 2*height//3]},
                {"name": "top", "box": [width//4, height//6, 3*width//4, height//3]},
                {"name": "bottom", "box": [width//4, 2*height//3, 3*width//4, 5*height//6]}
            ]
            
        return placement_zones
    
    def place_product(self, product_img, background_img, placement, scale=0.5, alpha=0.9):
        """Place product image on background according to placement information"""
        # Get placement area coordinates
        x1, y1, x2, y2 = placement["box"]
        placement_width = x2 - x1
        placement_height = y2 - y1
        
        # Resize product to fit the placement area
        target_height = int(placement_height * scale)
        ratio = product_img.width / product_img.height
        target_width = int(target_height * ratio)
        
        product_resized = product_img.resize((target_width, target_height))
        
        # Convert to numpy arrays for processing
        product_np = np.array(product_resized)
        background_np = background_img.copy()
        
        # Calculate position for centered placement
        pos_x = x1 + (placement_width - target_width) // 2
        pos_y = y1 + (placement_height - target_height) // 2
        
        # Ensure position is within image boundaries
        if pos_x < 0: pos_x = 0
        if pos_y < 0: pos_y = 0
        if pos_x + target_width > background_np.shape[1]: pos_x = background_np.shape[1] - target_width
        if pos_y + target_height > background_np.shape[0]: pos_y = background_np.shape[0] - target_height
        
        # Create mask from alpha channel
        if product_np.shape[2] == 4:  # RGBA
            mask = product_np[:, :, 3] / 255.0
            product_np = product_np[:, :, :3]
        else:
            mask = np.ones((target_height, target_width))
            
        # Apply the product to the background with alpha blending
        for c in range(3):  # RGB channels
            background_np[pos_y:pos_y+target_height, pos_x:pos_x+target_width, c] = \
                (1 - mask * alpha) * background_np[pos_y:pos_y+target_height, pos_x:pos_x+target_width, c] + \
                mask * alpha * product_np[:, :, c]
                
        return background_np
    
    def add_realistic_shadow(self, composite_img, pos_x, pos_y, width, height, opacity=0.3):
        """Add a simple shadow under the product to enhance realism"""
        # Create a copy of the image
        result = composite_img.copy()
        
        # Create an oval shadow shape
        shadow_width = int(width * 1.1)
        shadow_height = int(height * 0.2)
        shadow_y = pos_y + height - shadow_height//2
        
        # Make sure shadow is within bounds
        if pos_x + shadow_width > result.shape[1]:
            shadow_width = result.shape[1] - pos_x
        if shadow_y + shadow_height > result.shape[0]:
            shadow_height = result.shape[0] - shadow_y
            
        # Create shadow mask (oval shape)
        y, x = np.ogrid[:shadow_height, :shadow_width]
        mask = ((x - shadow_width/2) ** 2) / ((shadow_width/2) ** 2) + ((y - shadow_height/2) ** 2) / ((shadow_height/2) ** 2) <= 1
        
        # Apply shadow with fading
        for c in range(3):
            result[shadow_y:shadow_y+shadow_height, pos_x:pos_x+shadow_width, c][mask] = \
                result[shadow_y:shadow_y+shadow_height, pos_x:pos_x+shadow_width, c][mask] * (1 - opacity)
                
        return result
    
    def adjust_lighting(self, composite_img):
        """Adjust lighting to make the integration more realistic"""
        # Convert to HSV for easier light manipulation
        hsv = cv2.cvtColor(composite_img, cv2.COLOR_RGB2HSV)
        
        # Slightly enhance the value channel for better integration
        hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.05, 0, 255).astype(np.uint8)
        
        # Convert back to RGB
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return result
    
    def process_product(self, product_path, output_path, 
                       scale=0.5, alpha=0.9, 
                       product_type="home decor", room_type=None):
        """Process a single product image"""
        # Determine room type if not specified
        if room_type is None:
            for key, value in self.product_to_room.items():
                if key in product_type.lower():
                    room_type = value
                    break
            
            if room_type is None:
                room_type = "living_room"  # Default
        
        # Remove background from product
        product_no_bg = self.remove_background(product_path)
        
        # Load a suitable background
        background_np, detected_room_type = self.load_background(room_type)
        
        # Get placement zones
        placements = self.get_placement_zones(detected_room_type, background_np)
        
        # Choose a placement area (random for variety)
        placement = random.choice(placements)
        
        # Place product on background
        composite = self.place_product(product_no_bg, background_np, placement, scale, alpha)
        
        # Add shadow for realism
        x1, y1, x2, y2 = placement["box"]
        shadow_width = int((x2 - x1) * scale)
        shadow_height = int((y2 - y1) * scale)
        shadow_x = x1 + ((x2 - x1) - shadow_width) // 2
        shadow_y = y1
        composite = self.add_realistic_shadow(composite, shadow_x, shadow_y, shadow_width, shadow_height)
        
        # Adjust lighting for better integration
        composite = self.adjust_lighting(composite)
        
        # Save the final composite
        cv2.imwrite(output_path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            
        return {
            "product": os.path.basename(product_path),
            "room_type": detected_room_type,
            "placement": placement["name"],
            "output": output_path
        }
    
    def batch_process(self, product_dir, output_dir, 
                     scale=0.5, alpha=0.9, 
                     product_type="home decor", room_type=None):
        """Process multiple product images"""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs("backgrounds", exist_ok=True)
        
        # Check for background images, create them if they don't exist
        for room, bg_list in self.backgrounds.items():
            for bg_path in bg_list:
                os.makedirs(os.path.dirname(bg_path), exist_ok=True)
                if not os.path.exists(bg_path):
                    print(f"Creating placeholder background for {room}")
                    placeholder = np.ones((768, 1024, 3), dtype=np.uint8) * 240
                    cv2.imwrite(bg_path, placeholder)
        
        product_files = [f for f in os.listdir(product_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not product_files:
            print(f"No product images found in {product_dir}")
            return []
            
        print(f"Found {len(product_files)} product images to process")
        
        results = []
        for product_file in product_files:
            product_path = os.path.join(product_dir, product_file)
            output_path = os.path.join(output_dir, f"lifestyle_{product_file}")
            
            # Use product filename to infer product type if not specified
            current_product_type = product_type
            for item_type in self.product_to_room.keys():
                if item_type in product_file.lower():
                    current_product_type = item_type
                    break
            
            print(f"Processing {product_file} as {current_product_type}...")
            
            try:
                result = self.process_product(
                    product_path, output_path, 
                    scale, alpha,
                    current_product_type, room_type
                )
                results.append(result)
                print(f"Successfully processed {product_file}")
            except Exception as e:
                print(f"Error processing {product_file}: {str(e)}")
                continue
            
        print(f"Batch processing complete. Processed {len(results)} of {len(product_files)} images.")
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Product-to-Lifestyle Image Integration Tool")
    parser.add_argument("--product_dir", required=True, help="Directory containing product images")
    parser.add_argument("--output_dir", required=True, help="Directory to save output images")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale factor for product size (0.1-1.0)")
    parser.add_argument("--alpha", type=float, default=0.9, help="Alpha blending factor (0.1-1.0)")
    parser.add_argument("--product_type", default="home decor", 
                      help="Type of product for better placement")
    parser.add_argument("--room_type", default=None,
                      help="Type of room for background (living_room, bedroom, etc.)")
    
    args = parser.parse_args()
    
    integrator = ProductLifestyleIntegrator()
    results = integrator.batch_process(
        args.product_dir, 
        args.output_dir,
        args.scale,
        args.alpha,
        args.product_type,
        args.room_type
    )
    
    print(f"Processing complete. {len(results)} images integrated and saved to {args.output_dir}")