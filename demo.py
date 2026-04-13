"""
Quick Demo Script for Image Caption Generator
Creates test images and generates captions
"""

import requests
from PIL import Image, ImageDraw
import os
import time

API_URL = "http://localhost:8000"

def create_test_images():
    """Create various test images for demonstration."""
    print("Creating test images...")
    
    # Test 1: Outdoor scene with sky and ground
    img1 = Image.new('RGB', (400, 300), color=(135, 206, 235))
    draw1 = ImageDraw.Draw(img1)
    draw1.rectangle([0, 200, 400, 300], fill=(34, 139, 34))  # Green ground
    draw1.ellipse([300, 30, 370, 100], fill=(255, 255, 0))    # Sun
    draw1.polygon([150, 100, 120, 200, 180, 200], fill=(0, 100, 0))  # Tree
    draw1.rectangle([145, 200, 155, 230], fill=(139, 69, 19))  # Trunk
    img1.save('demo_outdoor.jpg')
    
    # Test 2: Indoor scene (warm colors)
    img2 = Image.new('RGB', (400, 300), color=(255, 228, 181))
    draw2 = ImageDraw.Draw(img2)
    draw2.rectangle([50, 150, 350, 280], fill=(139, 69, 19))  # Table
    draw2.rectangle([100, 80, 150, 150], fill=(255, 0, 0))    # Object
    draw2.rectangle([250, 100, 300, 150], fill=(0, 0, 255))   # Object
    img2.save('demo_indoor.jpg')
    
    # Test 3: Nature scene (mostly green)
    img3 = Image.new('RGB', (400, 300), color=(34, 139, 34))
    draw3 = ImageDraw.Draw(img3)
    draw3.rectangle([0, 0, 400, 100], fill=(135, 206, 235))  # Sky strip
    for i in range(5):
        draw3.ellipse([i*80, 100, i*80+60, 250], fill=(0, 100, 0))  # Trees
    img3.save('demo_nature.jpg')
    
    print("✓ Created 3 test images")


def generate_caption(image_path):
    """Generate caption for an image."""
    with open(image_path, 'rb') as f:
        response = requests.post(
            f"{API_URL}/caption",
            files={'file': (os.path.basename(image_path), f, 'image/jpeg')}
        )
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.json().get("detail", "Unknown error")}


def main():
    print("=" * 70)
    print("Image Caption Generator Demo")
    print("Architecture: CNN (ResNet-50) + LSTM")
    print("=" * 70)
    
    # Check if server is running
    try:
        response = requests.get(f"{API_URL}/")
        print(f"\n✓ Server is running: {response.json()['message']}")
    except:
        print("\n❌ Server is not running! Start it with: python main.py")
        return
    
    # Create test images
    create_test_images()
    
    # Generate captions
    test_images = ['demo_outdoor.jpg', 'demo_indoor.jpg', 'demo_nature.jpg']
    
    print("\n" + "=" * 70)
    print("Generating Captions")
    print("=" * 70)
    
    for img_path in test_images:
        print(f"\n📸 Image: {img_path}")
        print("-" * 70)
        
        start = time.time()
        result = generate_caption(img_path)
        elapsed = time.time() - start
        
        if 'error' not in result:
            print(f"Caption: {result['caption']}")
            print(f"Method: {result['generation_method']}")
            print(f"Time: {result['inference_time_ms']}ms")
        else:
            print(f"Error: {result['error']}")
    
    print("\n" + "=" * 70)
    print("✓ Demo Complete!")
    print("=" * 70)
    print("\nAPI Endpoints:")
    print("  GET  /         - Health check")
    print("  POST /caption  - Generate caption for single image")
    print("  POST /caption/batch - Generate captions for multiple images")
    print("  GET  /info     - Model information")
    print("\nTry uploading your own images to http://localhost:8000/caption")


if __name__ == "__main__":
    main()
