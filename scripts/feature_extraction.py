# scripts/feature_extraction.py
"""
İç Mekan Benzerlik Arama Sistemi - Özellik Çıkarımı
Manuel eklenen görsellerden özellik çıkarımı
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageStat
import pandas as pd
from tqdm import tqdm
import cv2
import joblib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json

# Ana dizini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Config bilgileri (config.py yoksa default değerler)
try:
    from config import *
except ImportError:
    print("⚠️ config.py bulunamadı, default ayarlar kullanılıyor...")
    DATA_DIR = "data"
    MODELS_DIR = "models" 
    RESULTS_DIR = "results"
    IMAGE_SIZE = (224, 224)
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']
    MIN_IMAGE_SIZE = (224, 224)
    PCA_COMPONENTS = 128
    BATCH_SIZE = 16

class FeatureExtractor:
    def __init__(self, device='auto'):
        """
        Özellik çıkarım sınıfı - manuel görsellerden özellik çıkarır
        """
        # Cihaz seçimi
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Kullanılan cihaz: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Modelleri yükle
        self._setup_models()
        self._setup_transforms()
    
    def _setup_models(self):
        """ResNet-50 modelini yükle"""
        print("📦 ResNet-50 modeli yükleniyor...")
        
        # ResNet-50 önceden eğitilmiş model
        self.resnet = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
        
        # Son classification katmanını kaldır (sadece features)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Evaluation moduna al
        self.resnet.eval()
        self.resnet.to(self.device)
        
        # Gradyan hesaplama kapalı (sadece inference)
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        print("✅ ResNet-50 hazır (2048 boyutlu özellik vektörü)")
    
    def _setup_transforms(self):
        """Görsel ön işleme dönüşümleri"""
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),  # 224x224'e yeniden boyutlandır
            transforms.ToTensor(),          # PIL -> Tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet ortalaması
                std=[0.229, 0.224, 0.225]    # ImageNet standart sapması
            )
        ])
        print("✅ Görsel dönüşümleri hazır")
    
    def validate_image(self, image_path):
        """Görselin kullanılabilir olup olmadığını kontrol et"""
        try:
            # Dosya var mı?
            if not os.path.exists(image_path):
                return False, "Dosya bulunamadı"
            
            # PIL ile açılabiliyor mu?
            img = Image.open(image_path).convert('RGB')
            width, height = img.size
            
            # Çok küçük mü?
            if width < MIN_IMAGE_SIZE[0] or height < MIN_IMAGE_SIZE[1]:
                return False, f"Çok küçük: {width}x{height}"
            
            # Çok aşırı aspect ratio?
            aspect_ratio = width / height
            if aspect_ratio > 10 or aspect_ratio < 0.1:
                return False, f"Aşırı en-boy oranı: {aspect_ratio:.2f}"
            
            # Bulanık mı? (OpenCV ile)
            try:
                cv_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if cv_img is not None:
                    blur_measure = cv2.Laplacian(cv_img, cv2.CV_64F).var()
                    if blur_measure < 10:  # Çok bulanık
                        return False, f"Çok bulanık: {blur_measure:.1f}"
            except:
                pass  # OpenCV hatası olursa geç
            
            # Renk çeşitliliği var mı?
            stat = ImageStat.Stat(img)
            if sum(stat.stddev) < 5:  # Tek renkli
                return False, f"Tek renkli/düşük kontrast"
            
            return True, "✅ Geçerli"
            
        except Exception as e:
            return False, f"Hata: {str(e)}"
    
    def extract_single_image_features(self, image_path):
        """Tek görselden özellik çıkar"""
        try:
            # Önce görseli validate et
            is_valid, reason = self.validate_image(image_path)
            if not is_valid:
                return None, reason
            
            # Görseli yükle ve dönüştür
            image = Image.open(image_path).convert('RGB')
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Özellik çıkarımı
            with torch.no_grad():
                features = self.resnet(tensor)
                features = features.squeeze().flatten()  # (2048,) boyutunda
            
            return features.cpu().numpy(), "✅ Başarılı"
            
        except Exception as e:
            return None, f"❌ Özellik çıkarım hatası: {str(e)}"
    
    def scan_data_directory(self):
        """data/ klasöründeki tüm görselleri tara"""
        print("🔍 Veri klasörü taranıyor...")
        
        categories = ['living_rooms', 'bedrooms', 'kitchens', 'bathrooms', 'dining_rooms']
        all_images = []
        
        for category in categories:
            category_path = os.path.join(DATA_DIR, category)
            
            if not os.path.exists(category_path):
                print(f"⚠️  Klasör bulunamadı: {category_path}")
                continue
            
            # Bu kategorideki dosyaları bul
            files = []
            for file in os.listdir(category_path):
                file_lower = file.lower()
                if any(file_lower.endswith(ext) for ext in SUPPORTED_FORMATS):
                    files.append(file)
            
            print(f"📁 {category}: {len(files)} dosya bulundu")
            
            # Dosya bilgilerini topla
            for filename in files:
                filepath = os.path.join(category_path, filename)
                try:
                    filesize = os.path.getsize(filepath)
                    all_images.append({
                        'filepath': filepath,
                        'filename': filename,
                        'category': category,
                        'filesize': filesize
                    })
                except:
                    print(f"  ⚠️ Dosya okunamadı: {filename}")
        
        print(f"\n📊 Toplam {len(all_images)} görsel bulundu")
        
        # Kategori özeti
        if all_images:
            df = pd.DataFrame(all_images)
            print("\n📈 Kategori Dağılımı:")
            category_counts = df['category'].value_counts()
            for category, count in category_counts.items():
                print(f"  📸 {category.replace('_', ' ').title()}: {count} görsel")
        
        return all_images
    
    def extract_features_from_list(self, image_list):
        """Görsel listesinden toplu özellik çıkarımı"""
        print(f"\n🧠 {len(image_list)} görselden özellik çıkarımı başlıyor...")
        
        all_features = []
        valid_metadata = []
        failed_count = 0
        
        # Progress bar ile her görseli işle
        for item in tqdm(image_list, desc="Özellik çıkarımı", unit="görsel"):
            features, message = self.extract_single_image_features(item['filepath'])
            
            if features is not None:
                all_features.append(features)
                valid_metadata.append(item)
            else:
                failed_count += 1
                # İlk 5 hatayı göster
                if failed_count <= 5:
                    print(f"\n  ❌ {item['filename']}: {message}")
        
        if not all_features:
            print("❌ Hiç özellik çıkarılamadı!")
            return None, None
        
        # NumPy array'e çevir
        features_array = np.array(all_features)
        metadata_df = pd.DataFrame(valid_metadata)
        
        print(f"\n✅ Özellik çıkarımı tamamlandı:")
        print(f"  🎯 Başarılı: {len(all_features)} görsel")
        print(f"  ❌ Başarısız: {failed_count} görsel")
        print(f"  📐 Özellik boyutu: {features_array.shape}")
        
        return features_array, metadata_df
    
    def apply_pca_reduction(self, features):
        """PCA ile boyut indirgeme"""
        print(f"\n📊 PCA boyut indirgeme ({PCA_COMPONENTS} bileşen)...")
        
        # PCA modelini eğit
        pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
        features_pca = pca.fit_transform(features)
        
        # Açıklanan varyans analizi
        explained_variance = np.sum(pca.explained_variance_ratio_)
        print(f"  📈 Açıklanan toplam varyans: {explained_variance*100:.1f}%")
        
        # PCA grafiği çiz
        self.plot_pca_analysis(pca)
        
        # PCA modelini kaydet
        pca_path = os.path.join(MODELS_DIR, 'pca_model.pkl')
        joblib.dump(pca, pca_path)
        print(f"  💾 PCA modeli kaydedildi: {pca_path}")
        
        return features_pca, pca
    
    def plot_pca_analysis(self, pca):
        """PCA analiz grafikleri"""
        plt.figure(figsize=(15, 5))
        
        # 1. Kümülatif açıklanan varyans
        plt.subplot(1, 3, 1)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        plt.plot(range(1, len(cumsum) + 1), cumsum, 'bo-', linewidth=2)
        plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='%95')
        plt.axhline(y=0.90, color='orange', linestyle='--', alpha=0.7, label='%90')
        plt.xlabel('Bileşen Sayısı')
        plt.ylabel('Kümülatif Açıklanan Varyans')
        plt.title('PCA Kümülatif Varyans')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 2. İlk 20 bileşenin bireysel varyansı
        plt.subplot(1, 3, 2)
        plt.bar(range(1, min(21, len(pca.explained_variance_ratio_) + 1)), 
                pca.explained_variance_ratio_[:20])
        plt.xlabel('Bileşen')
        plt.ylabel('Açıklanan Varyans')
        plt.title('İlk 20 Bileşen Varyansı')
        plt.xticks(range(1, min(21, len(pca.explained_variance_ratio_) + 1)))
        
        # 3. Varyans dağılımı
        plt.subplot(1, 3, 3)
        plt.semilogy(pca.explained_variance_ratio_, 'go-')
        plt.xlabel('Bileşen')
        plt.ylabel('Açıklanan Varyans (log)')
        plt.title('Varyans Dağılımı (Log Scale)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Kaydet
        plot_path = os.path.join(RESULTS_DIR, 'pca_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  📊 PCA analiz grafiği: {plot_path}")
        plt.show()
        
        # %95 varyans için gerekli bileşen sayısı
        variance_95_idx = np.argmax(cumsum >= 0.95) + 1
        variance_90_idx = np.argmax(cumsum >= 0.90) + 1
        print(f"  📍 %90 varyans: {variance_90_idx} bileşen")
        print(f"  📍 %95 varyans: {variance_95_idx} bileşen")
    
    def save_results(self, features, metadata_df, raw_features=None):
        """Sonuçları kaydet"""
        print("\n💾 Sonuçlar kaydediliyor...")
        
        # İndirgenen özellikler
        features_path = os.path.join(MODELS_DIR, 'features.npy')
        np.save(features_path, features)
        print(f"  ✅ Özellikler: {features_path}")
        
        # Ham özellikler (opsiyonel)
        if raw_features is not None:
            raw_features_path = os.path.join(MODELS_DIR, 'raw_features.npy')
            np.save(raw_features_path, raw_features)
            print(f"  ✅ Ham özellikler: {raw_features_path}")
        
        # Metadata
        metadata_path = os.path.join(MODELS_DIR, 'image_metadata.csv')
        metadata_df.to_csv(metadata_path, index=False, encoding='utf-8')
        print(f"  ✅ Metadata: {metadata_path}")
        
        # Dataset istatistikleri
        stats = {
            'extraction_date': pd.Timestamp.now().isoformat(),
            'total_images': len(metadata_df),
            'feature_dimension': features.shape[1],
            'raw_feature_dimension': raw_features.shape[1] if raw_features is not None else 'N/A',
            'categories': metadata_df['category'].value_counts().to_dict(),
            'file_sizes': {
                'min_mb': metadata_df['filesize'].min() / 1024**2,
                'max_mb': metadata_df['filesize'].max() / 1024**2,
                'avg_mb': metadata_df['filesize'].mean() / 1024**2
            }
        }
        
        stats_path = os.path.join(MODELS_DIR, 'dataset_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"  📊 İstatistikler: {stats_path}")
        
        return features_path, metadata_path, stats_path
    
    def run_complete_pipeline(self):
        """Tam özellik çıkarım işlemi"""
        print("🚀 İç Mekan Özellik Çıkarım Pipeline'ı")
        print("=" * 60)
        
        # Gerekli klasörleri oluştur
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        try:
            # 1. Veri tarama
            image_list = self.scan_data_directory()
            
            if not image_list:
                print("❌ Hiç görsel bulunamadı!")
                print("\n💡 Çözüm:")
                print("   1. data/ klasörüne şu alt klasörleri oluşturun:")
                print("      - living_rooms/")
                print("      - bedrooms/") 
                print("      - kitchens/")
                print("      - bathrooms/")
                print("      - dining_rooms/")
                print("   2. Her klasöre en az birkaç JPG/PNG görsel ekleyin")
                return False
            
            # 2. Özellik çıkarımı
            raw_features, metadata_df = self.extract_features_from_list(image_list)
            
            if raw_features is None:
                print("❌ Özellik çıkarımı başarısız!")
                return False
            
            # 3. PCA boyut indirgeme
            reduced_features, pca_model = self.apply_pca_reduction(raw_features)
            
            # 4. Sonuçları kaydet
            features_path, metadata_path, stats_path = self.save_results(
                reduced_features, metadata_df, raw_features
            )
            
            # 5. Başarı raporu
            print("\n🎉 Özellik Çıkarımı Başarıyla Tamamlandı!")
            print("=" * 60)
            print(f"📊 İşlenen görseller: {len(metadata_df)}")
            print(f"📐 Ham özellik boyutu: 2048")
            print(f"📉 PCA özellik boyutu: {reduced_features.shape[1]}")
            print(f"💾 Kaydedilen dosyalar:")
            print(f"   • {features_path}")
            print(f"   • {metadata_path}")
            print(f"   • {os.path.join(MODELS_DIR, 'pca_model.pkl')}")
            print(f"   • {stats_path}")
            print("\n🚀 Sonraki adım:")
            print("   python scripts/similarity_search.py")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Pipeline hatası: {e}")
            import traceback
            traceback.print_exc()
            return False

def check_data_directory():
    """Data klasör yapısını kontrol et"""
    print("🔍 Veri klasörü kontrolü...")
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ Ana veri klasörü bulunamadı: {DATA_DIR}")
        return False
    
    categories = ['living_rooms', 'bedrooms', 'kitchens', 'bathrooms', 'dining_rooms']
    missing_categories = []
    empty_categories = []
    
    for category in categories:
        category_path = os.path.join(DATA_DIR, category)
        
        if not os.path.exists(category_path):
            missing_categories.append(category)
        else:
            files = [f for f in os.listdir(category_path) 
                    if any(f.lower().endswith(ext) for ext in SUPPORTED_FORMATS)]
            if len(files) == 0:
                empty_categories.append(category)
            else:
                print(f"  ✅ {category}: {len(files)} görsel")
    
    # Eksik klasörler
    if missing_categories:
        print(f"\n❌ Eksik klasörler:")
        for cat in missing_categories:
            print(f"   • {os.path.join(DATA_DIR, cat)}")
    
    # Boş klasörler  
    if empty_categories:
        print(f"\n⚠️ Boş klasörler:")
        for cat in empty_categories:
            print(f"   • {os.path.join(DATA_DIR, cat)}")
    
    if missing_categories or empty_categories:
        print(f"\n💡 Her klasöre en az birkaç görsel ekleyin:")
        print(f"   Desteklenen formatlar: {', '.join(SUPPORTED_FORMATS)}")
        return False
    
    print("✅ Veri klasörü yapısı uygun!")
    return True

def main():
    """Ana fonksiyon"""
    print("🧠 İç Mekan Görsel Özellik Çıkarımı")
    print("=" * 50)
    
    # Veri klasörünü kontrol et
    if not check_data_directory():
        print("\n🔧 Önce veri klasörlerini düzenleyin!")
        return False
    
    # Mevcut özellikleri kontrol et
    existing_features = os.path.join(MODELS_DIR, 'features.npy')
    existing_metadata = os.path.join(MODELS_DIR, 'image_metadata.csv')
    
    if os.path.exists(existing_features) and os.path.exists(existing_metadata):
        print(f"\n🔄 Mevcut özellikler bulundu!")
        print(f"   • {existing_features}")
        print(f"   • {existing_metadata}")
        
        choice = input("\nYeniden çıkarım yapmak istiyor musunuz? (y/N): ").lower().strip()
        if choice != 'y':
            print("✅ Mevcut özellikler kullanılacak")
            print("🚀 Sonraki adım: python scripts/similarity_search.py")
            return True
    
    # Özellik çıkarımı başlat
    extractor = FeatureExtractor()
    success = extractor.run_complete_pipeline()
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)