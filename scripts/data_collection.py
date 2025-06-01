# scripts/data_collection.py
"""
İç Mekan Benzerlik Arama Sistemi - Veri Toplama
Unsplash ve Pexels API'leri kullanarak görsel indirme
"""

import os
import sys
import requests
from PIL import Image
import time
import json
from tqdm import tqdm

# Ana dizini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class DataCollector:
    def __init__(self):
        self.unsplash_key = UNSPLASH_ACCESS_KEY
        self.pexels_key = PEXELS_API_KEY
        self.categories = CATEGORIES
        self.data_dir = DATA_DIR
        
        # Veri klasörlerini oluştur
        for category in self.categories.keys():
            os.makedirs(os.path.join(self.data_dir, category), exist_ok=True)
    
    def download_image(self, url, save_path, timeout=10):
        """URL'den görsel indirme"""
        try:
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Görselin geçerli olduğunu kontrol et
            img = Image.open(save_path)
            width, height = img.size
            
            # Minimum boyut kontrolü
            if width < MIN_IMAGE_SIZE[0] or height < MIN_IMAGE_SIZE[1]:
                os.remove(save_path)
                return False, "Çok küçük boyut"
            
            # Dosya boyutu kontrolü
            file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                os.remove(save_path)
                return False, "Dosya çok büyük"
            
            img.verify()
            return True, "Başarılı"
        
        except Exception as e:
            if os.path.exists(save_path):
                os.remove(save_path)
            return False, str(e)
    
    def search_unsplash_images(self, query, count=30):
        """Unsplash API ile görsel arama"""
        if not self.unsplash_key or self.unsplash_key == "YOUR_UNSPLASH_API_KEY_HERE":
            return []
        
        url = "https://api.unsplash.com/search/photos"
        headers = {"Authorization": f"Client-ID {self.unsplash_key}"}
        
        params = {
            "query": query,
            "per_page": min(count, 30),
            "orientation": "landscape",
            "content_filter": "high"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            urls = []
            for photo in data.get('results', []):
                urls.append({
                    'url': photo['urls']['regular'],
                    'id': photo['id'],
                    'description': photo.get('description', ''),
                    'author': photo['user']['name']
                })
            
            return urls
        
        except Exception as e:
            print(f"❌ Unsplash API hatası: {e}")
            return []
    
    def search_pexels_images(self, query, count=30):
        """Pexels API ile görsel arama"""
        if not self.pexels_key or self.pexels_key == "YOUR_PEXELS_API_KEY_HERE":
            return []
        
        url = "https://api.pexels.com/v1/search"
        headers = {"Authorization": self.pexels_key}
        
        params = {
            "query": query,
            "per_page": min(count, 80),
            "orientation": "landscape"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            urls = []
            for photo in data.get('photos', []):
                urls.append({
                    'url': photo['src']['large'],
                    'id': photo['id'],
                    'description': photo.get('alt', ''),
                    'author': photo['photographer']
                })
            
            return urls
        
        except Exception as e:
            print(f"❌ Pexels API hatası: {e}")
            return []
    
    def create_test_images(self):
        """Test görselleri oluştur (API anahtarı yoksa)"""
        print("🎨 Test görselleri oluşturuluyor...")
        
        # Her kategori için farklı renkler
        colors = [
            (220, 150, 150),  # living_rooms - açık kırmızı
            (150, 220, 150),  # bedrooms - açık yeşil
            (150, 150, 220),  # kitchens - açık mavi
            (220, 220, 150),  # bathrooms - açık sarı
            (220, 150, 220)   # dining_rooms - açık mor
        ]
        
        patterns = ['solid', 'gradient', 'checkerboard', 'stripes']
        
        for i, category in enumerate(self.categories.keys()):
            category_path = os.path.join(self.data_dir, category)
            base_color = colors[i]
            
            for j in range(50):  # Her kategoriden 50 test görseli
                # Farklı desenler oluştur
                pattern = patterns[j % len(patterns)]
                
                if pattern == 'solid':
                    img = Image.new('RGB', IMAGE_SIZE, base_color)
                
                elif pattern == 'gradient':
                    img = Image.new('RGB', IMAGE_SIZE, base_color)
                    # Basit gradient efekti
                    for x in range(IMAGE_SIZE[0]):
                        for y in range(IMAGE_SIZE[1]):
                            factor = x / IMAGE_SIZE[0]
                            new_color = tuple(int(c * factor) for c in base_color)
                            img.putpixel((x, y), new_color)
                
                elif pattern == 'checkerboard':
                    img = Image.new('RGB', IMAGE_SIZE, base_color)
                    darker_color = tuple(max(0, c - 50) for c in base_color)
                    for x in range(0, IMAGE_SIZE[0], 20):
                        for y in range(0, IMAGE_SIZE[1], 20):
                            if (x // 20 + y // 20) % 2:
                                for dx in range(20):
                                    for dy in range(20):
                                        if x + dx < IMAGE_SIZE[0] and y + dy < IMAGE_SIZE[1]:
                                            img.putpixel((x + dx, y + dy), darker_color)
                
                else:  # stripes
                    img = Image.new('RGB', IMAGE_SIZE, base_color)
                    darker_color = tuple(max(0, c - 30) for c in base_color)
                    for y in range(0, IMAGE_SIZE[1], 10):
                        if (y // 10) % 2:
                            for x in range(IMAGE_SIZE[0]):
                                for dy in range(min(10, IMAGE_SIZE[1] - y)):
                                    img.putpixel((x, y + dy), darker_color)
                
                save_path = os.path.join(category_path, f"test_{j:03d}.jpg")
                img.save(save_path, 'JPEG', quality=85)
            
            print(f"  ✅ {category}: 50 test görseli oluşturuldu")
    
    def collect_real_images(self):
        """Gerçek görselleri API'lerden topla"""
        print("🌐 Gerçek görseller API'lerden indiriliyor...")
        
        total_downloaded = 0
        
        for category, queries in self.categories.items():
            print(f"\n📁 {category.replace('_', ' ').title()} kategorisi...")
            category_path = os.path.join(self.data_dir, category)
            
            img_count = 0
            target_count = MAX_IMAGES_PER_CATEGORY
            
            for query in queries:
                if img_count >= target_count:
                    break
                
                print(f"  🔍 Arama: '{query}'")
                
                # Unsplash'dan ara
                unsplash_urls = self.search_unsplash_images(query, count=20)
                
                # Pexels'den ara
                pexels_urls = self.search_pexels_images(query, count=20)
                
                # Tüm URL'leri birleştir
                all_urls = unsplash_urls + pexels_urls
                
                # İndir
                for item in tqdm(all_urls, desc=f"  İndiriliyor", leave=False):
                    if img_count >= target_count:
                        break
                    
                    save_path = os.path.join(category_path, f"{category}_{img_count:03d}.jpg")
                    success, message = self.download_image(item['url'], save_path)
                    
                    if success:
                        img_count += 1
                        total_downloaded += 1
                        
                        # Metadata kaydet
                        metadata = {
                            'filename': os.path.basename(save_path),
                            'source_url': item['url'],
                            'source_id': item['id'],
                            'description': item['description'],
                            'author': item['author'],
                            'query': query,
                            'category': category
                        }
                        
                        metadata_path = save_path.replace('.jpg', '_metadata.json')
                        with open(metadata_path, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, indent=2, ensure_ascii=False)
                    
                    time.sleep(0.5)  # Rate limiting
                
                time.sleep(1)  # Query arası bekleme
            
            print(f"  ✅ {category}: {img_count} görsel indirildi")
        
        print(f"\n🎉 Toplam {total_downloaded} görsel başarıyla indirildi!")
    
    def get_download_statistics(self):
        """İndirme istatistiklerini göster"""
        print("\n📊 Veri Kümesi İstatistikleri:")
        print("=" * 40)
        
        total_images = 0
        for category in self.categories.keys():
            category_path = os.path.join(self.data_dir, category)
            if os.path.exists(category_path):
                files = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                count = len(files)
                total_images += count
                print(f"📁 {category.replace('_', ' ').title()}: {count} görsel")
        
        print(f"\n📈 Toplam: {total_images} görsel")
        print(f"💾 Ortalama kategori başına: {total_images / len(self.categories):.1f} görsel")
        
        return total_images
    
    def run_data_collection(self):
        """Ana veri toplama fonksiyonu"""
        print("📥 İç Mekan Veri Toplama Başlıyor...")
        print("=" * 50)
        
        # API anahtarlarını kontrol et
        has_api_keys = (
            self.unsplash_key != "YOUR_UNSPLASH_API_KEY_HERE" or 
            self.pexels_key != "YOUR_PEXELS_API_KEY_HERE"
        )
        
        if has_api_keys:
            print("🔑 API anahtarları bulundu, gerçek görseller indiriliyor...")
            self.collect_real_images()
        else:
            print("⚠️  API anahtarı bulunamadı, test görselleri oluşturuluyor...")
            print("💡 Gerçek görseller için config.py dosyasında API anahtarlarını güncelleyin")
            self.create_test_images()
        
        # İstatistikleri göster
        total = self.get_download_statistics()
        
        if total > 0:
            print("\n✅ Veri toplama başarıyla tamamlandı!")
        else:
            print("\n❌ Veri toplama başarısız oldu!")
        
        return total > 0

if __name__ == "__main__":
    collector = DataCollector()
    success = collector.run_data_collection()
    
    if success:
        print("\n🚀 Sonraki adım: Özellik çıkarımı")
        print("   python scripts/feature_extraction.py")
    else:
        print("\n🔧 Veri toplama sorunlarını giderin ve tekrar deneyin")