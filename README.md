 
# Product-to-Lifestyle Image Integration Tool

This tool automatically places e-commerce product images into realistic lifestyle backgrounds using computer vision and generative AI techniques. It processes multiple product images in batch mode and seamlessly integrates them into diverse lifestyle scenes with realistic lighting, perspective, and scale.

## Features

- **Background Removal**: Automatically removes backgrounds from product images
- **Smart Placement Detection**: Identifies suitable surfaces in background images for product placement
- **Realistic Integration**: Places products with proper scaling and positioning
- **Harmonization**: Optional AI-powered harmonization for seamless lighting and style matching
- **Batch Processing**: Handles multiple product images simultaneously
- **Customization Options**: Adjustable scale, placement, and blending parameters

## Hardware Requirements

- **CPU**: Any modern multi-core CPU (Intel i5/Ryzen 5 or better recommended)
- **RAM**: Minimum 8GB, 16GB+ recommended
- **GPU**: Optional but recommended for faster processing
  - NVIDIA GPU with 4GB+ VRAM for acceleration
  - CUDA compatible for optimal performance
- **Storage**: 5GB for models and dependencies

## Software Requirements

All software and dependencies are free and open-source:

- Python 3.8 or higher
- PyTorch
- OpenCV
- Hugging Face Transformers
- Diffusers
- rembg
- PIL (Pillow)
- numpy

## Installation

1. Clone this repository:
```bash
git clone https://github.com/username/product-lifestyle-integration.git
cd product-lifestyle-integration
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
pip install transformers diffusers rembg opencv-python pillow numpy
```

How to Use It

Install the required dependencies:

```bash
pip install numpy opencv-python torch pillow rembg
```
Create your directory structure:

```bash
mkdir -p product_images output backgrounds
```

Place your product images in the product_images directory
Run the script:

```bash
python integrate.py --product_dir product_images --output_dir output
```






### Parameters

- `--product_dir`: Directory containing product images
- `--background_dir`: Directory containing background lifestyle images
- `--output_dir`: Directory where integrated images will be saved
- `--scale`: Scale factor for product size (0.1-1.0, default: 0.5)
- `--alpha`: Alpha blending factor (0.1-1.0, default: 0.8)
- `--no_harmonize`: Disable harmonization with Stable Diffusion
- `--product_type`: Type of product for better prompt engineering (default: "home decor")
- `--cpu`: Force CPU usage even if GPU is available

## How It Works

1. **Background Removal**: The tool uses rembg to remove backgrounds from product images, preserving transparency.

2. **Placement Detection**: It uses an object detection model to identify suitable surfaces (tables, shelves, counters, etc.) in the background images.

3. **Product Placement**: The product is scaled appropriately and placed on the detected surface with proper alpha blending.

4. **Harmonization (Optional)**: Stable Diffusion's img2img pipeline is used to harmonize the composite image, ensuring lighting and style consistency.

5. **Batch Processing**: Multiple product-background pairs are processed sequentially, with results saved to the output directory.

## Technical Approach

### Models Used

- **Background Removal**: rembg (based on U2Net)
- **Object Detection**: Microsoft's Table Transformer (DETR) for detecting suitable placement surfaces
- **Image Harmonization**: Stable Diffusion v1.4 (CompVis/stable-diffusion-v1-4)

### Advantages

- **Fully Open-Source**: Uses only free, unlimited-usage libraries and models
- **No External API Dependencies**: All processing happens locally
- **Customizable**: Adjustable parameters for different product types and integration styles
- **Realistic Results**: Multi-stage pipeline ensures natural-looking placement

## Limitations and Considerations

- Processing time depends on hardware capabilities (GPU significantly improves performance)
- Large or complex images may require more memory
- Some complex product shapes may not be perfectly segmented
- The harmonization step, while improving realism, may slightly alter product details

## Sample Results
![integrated_471034853001-0441_composite](https://github.com/user-attachments/assets/3965a4b3-109d-4e99-a17e-a1b4fbd0de6d)

The tool has been tested with various product types:
- Decorative vases on coffee tables and shelves
- Modern lamps on bedside tables and desks
- Cushions and throws on sofas and beds
- Small appliances on kitchen counters

Check the `examples` directory for sample inputs and outputs.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
