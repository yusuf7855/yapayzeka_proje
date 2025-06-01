# config.py
"""
İç Mekan Benzerlik Arama Sistemi - Genel Ayarlar
"""

import os

# API Anahtarları
UNSPLASH_ACCESS_KEY = "YOUR_UNSPLASH_API_KEY_HERE"
PEXELS_API_KEY = "YOUR_PEXELS_API_KEY_HERE"

# Proje Ayarları
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "test_images")
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "uploads")

# Kategori Tanımları
CATEGORIES = {
    'living_rooms': ['modern living room', 'cozy living room', 'minimalist living room'],
    'bedrooms': ['modern bedroom', 'cozy bedroom', 'minimalist bedroom'],
    'kitchens': ['modern kitchen', 'rustic kitchen', 'white kitchen'],
    'bathrooms': ['modern bathroom', 'luxury bathroom', 'small bathroom'],
    'dining_rooms': ['dining room', 'dining table', 'kitchen dining']
}

# Model Ayarları
FEATURE_DIMENSION = 2048  # ResNet-50 çıkış boyutu
PCA_COMPONENTS = 128
UMAP_COMPONENTS = 64
KMEANS_CLUSTERS = 20

# Görsel Ayarları
IMAGE_SIZE = (224, 224)
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']

# Sistem Ayarları
BATCH_SIZE = 32
MAX_IMAGES_PER_CATEGORY = 100
MIN_IMAGE_SIZE = (224, 224)
MAX_FILE_SIZE_MB = 10

# Flask Ayarları
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True

# Logging Ayarları
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'