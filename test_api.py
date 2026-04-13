"""
Test script for the Image Caption Generator
"""

import requests
import os
from PIL import Image
import io

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("\n1. Testing health endpoint...")
    response = requests.get(f"{API_URL}/")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    return response.status_code == 200

def test_caption_upload(image_path):
    """Test single image caption endpoint."""
    print(f"\n2. Testing caption generation for: {image_path}")
    
    with open(image_path, 'rb') as f:
        files = {'file': ('test.jpg', f, 'image/jpeg')}
        response = requests.post(f"{API_URL}/caption", files=files)
    
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Caption: {data['caption']}")
        print(f"   Inference time: {data['inference_time_ms']}ms")
        print(f"   Model: {data['model']}")
    else:
        print(f"   Error: {response.json()}")
    
    return response.status_code == 200

def test_model_info():
    """Test model info endpoint."""
    print("\n3. Testing model info endpoint...")
    response = requests.get(f"{API_URL}/info")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    return response.status_code == 200

def create_test_image(path="test_image.jpg"):
    """Create a simple test image."""
    print(f"\nCreating test image: {path}")
    img = Image.new('RGB', (224, 224), color=(73, 109, 137))
    img.save(path)
    print(f"✓ Test image created")
    return path

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Image Caption Generator API")
    print("=" * 60)
    
    # Test health
    if not test_health():
        print("\n❌ Health check failed. Is the server running?")
        exit(1)
    
    # Create test image
    test_img_path = create_test_image()
    
    # Test caption generation
    test_caption_upload(test_img_path)
    
    # Test model info
    test_model_info()
    
    # Cleanup
    if os.path.exists(test_img_path):
        os.remove(test_img_path)
    
    print("\n" + "=" * 60)
    print("✓ All tests completed!")
    print("=" * 60)
