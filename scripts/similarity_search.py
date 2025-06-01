# scripts/similarity_search.py
"""
İç Mekan Benzerlik Arama Sistemi - Benzerlik Arama
Özellik vektörleri kullanarak görsel benzerlik arama
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
import joblib
import json
from datetime import datetime
import time

# Ana dizini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class SimilaritySearcher:
    def __init__(self, load_models=True):
        """
        Benzerlik arama sınıfı
        Args:
            load_models: Modelleri otomatik yükle
        """
        self.features = None
        self.metadata = None
        self.pca_model = None
        self.umap_model = None
        self.kmeans_model = None
        self.style_clusters = None
        
        if load_models:
            self.load_models()
    
    def load_models(self):
        """Kaydedilmiş modelleri ve verileri yükle"""
        print("📦 Modeller ve veriler yükleniyor...")
        
        # Özellik vektörlerini yükle
        features_path = os.path.join(MODELS_DIR, 'features.npy')
        if os.path.exists(features_path):
            self.features = np.load(features_path)
            print(f"  ✅ Özellikler yüklendi: {self.features.shape}")
        else:
            raise FileNotFoundError(f"Özellik dosyası bulunamadı: {features_path}")
        
        # Metadata yükle
        metadata_path = os.path.join(MODELS_DIR, 'image_metadata.csv')
        if os.path.exists(metadata_path):
            self.metadata = pd.read_csv(metadata_path)
            print(f"  ✅ Metadata yüklendi: {len(self.metadata)} görsel")
        else:
            raise FileNotFoundError(f"Metadata dosyası bulunamadı: {metadata_path}")
        
        # PCA modeli yükle
        pca_path = os.path.join(MODELS_DIR, 'pca_model.pkl')
        if os.path.exists(pca_path):
            self.pca_model = joblib.load(pca_path)
            print(f"  ✅ PCA modeli yüklendi")
        
        # UMAP modeli yükle (opsiyonel)
        umap_path = os.path.join(MODELS_DIR, 'umap_model.pkl')
        if os.path.exists(umap_path):
            self.umap_model = joblib.load(umap_path)
            print(f"  ✅ UMAP modeli yüklendi")
        
        # Stil kümeleri oluştur
        self._create_style_clusters()
        
        print("🎯 Benzerlik arama sistemi hazır!")
    
    def _create_style_clusters(self):
        """Stil kümeleri oluştur"""
        print("  🎨 Stil kümeleri oluşturuluyor...")
        
        # K-Means kümeleme uygula
        self.kmeans_model = KMeans(n_clusters=KMEANS_CLUSTERS, random_state=42, n_init=10)
        self.style_clusters = self.kmeans_model.fit_predict(self.features)
        
        # Kümeleme modelini kaydet
        kmeans_path = os.path.join(MODELS_DIR, 'kmeans_model.pkl')
        joblib.dump(self.kmeans_model, kmeans_path)
        
        # Küme istatistikleri
        unique_clusters, cluster_counts = np.unique(self.style_clusters, return_counts=True)
        print(f"  📊 {len(unique_clusters)} stil kümesi oluşturuldu")
        
        # Küme-kategori ilişkisi
        cluster_category_map = {}
        for cluster_id in unique_clusters:
            cluster_mask = (self.style_clusters == cluster_id)
            cluster_categories = self.metadata[cluster_mask]['category'].value_counts()
            dominant_category = cluster_categories.index[0] if len(cluster_categories) > 0 else 'unknown'
            cluster_category_map[cluster_id] = {
                'dominant_category': dominant_category,
                'size': int(cluster_counts[cluster_id]),
                'categories': cluster_categories.to_dict()
            }
        
        # Küme bilgilerini kaydet
        cluster_info_path = os.path.join(MODELS_DIR, 'cluster_info.json')
        with open(cluster_info_path, 'w', encoding='utf-8') as f:
            json.dump(cluster_category_map, f, indent=2, ensure_ascii=False)
        
        print(f"  💾 Küme bilgileri kaydedildi: {cluster_info_path}")
    
    def extract_query_features(self, query_image_path):
        """Sorgu görseli için özellik çıkarımı"""
        # Feature extraction modülünü import et
        from feature_extraction import FeatureExtractor
        
        extractor = FeatureExtractor()
        raw_features, message = extractor.extract_single_image_features(query_image_path)
        
        if raw_features is None:
            return None, message
        
        # Boyut indirgeme uygula
        if self.pca_model is not None:
            # PCA uygula
            features_pca = self.pca_model.transform([raw_features])
            
            # UMAP uygula (varsa)
            if self.umap_model is not None:
                final_features = self.umap_model.transform(features_pca)
            else:
                final_features = features_pca
        else:
            final_features = raw_features.reshape(1, -1)
        
        return final_features[0], "Başarılı"
    
    def calculate_similarity_scores(self, query_features, method='cosine'):
        """Benzerlik puanlarını hesapla"""
        if method == 'cosine':
            similarities = cosine_similarity([query_features], self.features)[0]
        elif method == 'euclidean':
            distances = euclidean_distances([query_features], self.features)[0]
            # Mesafeyi benzerliğe çevir (0-1 arası normalize et)
            max_dist = np.max(distances)
            similarities = 1 - (distances / max_dist)
        else:
            raise ValueError(f"Desteklenmeyen benzerlik metodu: {method}")
        
        return similarities
    
    def search_similar_images(self, query_image_path, k=10, category_filter=None, 
                            similarity_method='cosine', min_similarity=0.0):
        """
        Benzer görselleri ara
        
        Args:
            query_image_path: Sorgu görseli yolu
            k: Döndürülecek benzer görsel sayısı
            category_filter: Kategori filtresi (liste veya string)
            similarity_method: Benzerlik hesaplama yöntemi
            min_similarity: Minimum benzerlik eşiği
        """
        start_time = time.time()
        
        print(f"🔍 Benzer görseller aranıyor: {os.path.basename(query_image_path)}")
        
        # 1. Sorgu görseli özelliklerini çıkar
        query_features, message = self.extract_query_features(query_image_path)
        if query_features is None:
            return [], [], f"Özellik çıkarım hatası: {message}"
        
        # 2. Sorgu görseli stil kümesini bul
        query_cluster = self.kmeans_model.predict([query_features])[0]
        print(f"  🎨 Sorgu stil kümesi: {query_cluster}")
        
        # 3. Filtreleme uygula
        valid_indices = np.arange(len(self.features))
        
        if category_filter is not None:
            if isinstance(category_filter, str):
                category_filter = [category_filter]
            category_mask = self.metadata['category'].isin(category_filter)
            valid_indices = valid_indices[category_mask]
            print(f"  📂 Kategori filtresi uygulandı: {len(valid_indices)} görsel")
        
        if len(valid_indices) == 0:
            return [], [], "Filtre kriterlerine uygun görsel bulunamadı"
        
        # 4. Benzerlik hesapla
        filtered_features = self.features[valid_indices]
        similarities = self.calculate_similarity_scores(query_features, similarity_method)
        filtered_similarities = similarities[valid_indices]
        
        # 5. Minimum benzerlik filtresi
        if min_similarity > 0:
            similarity_mask = filtered_similarities >= min_similarity
            valid_indices = valid_indices[similarity_mask]
            filtered_similarities = filtered_similarities[similarity_mask]
            print(f"  📊 Minimum benzerlik filtresi: {len(valid_indices)} görsel")
        
        if len(valid_indices) == 0:
            return [], [], f"Minimum benzerlik ({min_similarity}) kriterini karşılayan görsel bulunamadı"
        
        # 6. En benzer k görseli seç
        k = min(k, len(valid_indices))
        top_indices = np.argsort(filtered_similarities)[::-1][:k]
        final_indices = valid_indices[top_indices]
        final_scores = filtered_similarities[top_indices]
        
        search_time = time.time() - start_time
        print(f"  ⚡ Arama süresi: {search_time:.2f} saniye")
        print(f"  🎯 En benzer {k} görsel bulundu")
        
        return final_indices, final_scores, "Başarılı"
    
    def get_similar_by_style(self, query_image_path, k=10):
        """Aynı stildeki benzer görselleri bul"""
        query_features, message = self.extract_query_features(query_image_path)
        if query_features is None:
            return [], [], message
        
        # Sorgu stil kümesini bul
        query_cluster = self.kmeans_model.predict([query_features])[0]
        
        # Aynı kümeden görselleri filtrele
        same_style_mask = (self.style_clusters == query_cluster)
        same_style_indices = np.where(same_style_mask)[0]
        
        if len(same_style_indices) == 0:
            return [], [], "Aynı stilde görsel bulunamadı"
        
        # Benzerlik hesapla
        similarities = self.calculate_similarity_scores(query_features)
        same_style_similarities = similarities[same_style_indices]
        
        # En benzer k görseli seç
        k = min(k, len(same_style_indices))
        top_indices = np.argsort(same_style_similarities)[::-1][:k]
        final_indices = same_style_indices[top_indices]
        final_scores = same_style_similarities[top_indices]
        
        return final_indices, final_scores, "Başarılı"
    
    def visualize_search_results(self, query_image_path, indices, scores, 
                               save_path=None, show_metadata=True):
        """Arama sonuçlarını görselleştir"""
        n_results = len(indices)
        n_cols = min(5, n_results + 1)
        n_rows = (n_results + n_cols) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Sorgu görseli
        try:
            query_img = Image.open(query_image_path)
            axes[0, 0].imshow(query_img)
            axes[0, 0].set_title("SORGU GÖRSELİ", fontweight='bold', fontsize=10)
            axes[0, 0].axis('off')
        except Exception as e:
            axes[0, 0].text(0.5, 0.5, f"Görsel yüklenemedi\n{str(e)}", 
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].axis('off')
        
        # Benzer görseller
        for i, (idx, score) in enumerate(zip(indices, scores)):
            row = (i + 1) // n_cols
            col = (i + 1) % n_cols
            
            if row >= n_rows:
                break
            
            try:
                result_path = self.metadata.iloc[idx]['filepath']
                result_img = Image.open(result_path)
                axes[row, col].imshow(result_img)
                
                # Başlık bilgileri
                title = f"#{i+1} - {score*100:.1f}%"
                if show_metadata:
                    category = self.metadata.iloc[idx]['category']
                    cluster = self.style_clusters[idx]
                    title += f"\n{category.replace('_', ' ').title()}\nKüme: {cluster}"
                
                axes[row, col].set_title(title, fontsize=8)
                axes[row, col].axis('off')
                
            except Exception as e:
                axes[row, col].text(0.5, 0.5, f"Görsel yüklenemedi\n{str(e)}", 
                                   ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].axis('off')
        
        # Boş eksenleri gizle
        for i in range(n_results + 1, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if row < n_rows and col < n_cols:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # Kaydet
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(RESULTS_DIR, f"search_results_{timestamp}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Sonuçlar kaydedildi: {save_path}")
        
        plt.show()
        return save_path
    
    def generate_search_report(self, query_image_path, indices, scores, 
                             search_params=None):
        """Detaylı arama raporu oluştur"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'query_image': os.path.basename(query_image_path),
            'search_params': search_params or {},
            'results_count': len(indices),
            'results': []
        }
        
        # Sonuç detayları
        for i, (idx, score) in enumerate(zip(indices, scores)):
            result_info = {
                'rank': i + 1,
                'similarity_score': float(score),
                'image_path': self.metadata.iloc[idx]['filepath'],
                'filename': self.metadata.iloc[idx]['filename'],
                'category': self.metadata.iloc[idx]['category'],
                'style_cluster': int(self.style_clusters[idx]),
                'file_size_mb': self.metadata.iloc[idx]['filesize'] / 1024**2
            }
            report['results'].append(result_info)
        
        # İstatistikler
        if len(scores) > 0:
            report['statistics'] = {
                'avg_similarity': float(np.mean(scores)),
                'max_similarity': float(np.max(scores)),
                'min_similarity': float(np.min(scores)),
                'std_similarity': float(np.std(scores))
            }
        
        # Kategori dağılımı
        if len(indices) > 0:
            result_categories = [self.metadata.iloc[idx]['category'] for idx in indices]
            category_counts = pd.Series(result_categories).value_counts()
            report['category_distribution'] = category_counts.to_dict()
        
        return report
    
    def save_search_report(self, report, filepath=None):
        """Arama raporunu JSON olarak kaydet"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(RESULTS_DIR, f"search_report_{timestamp}.json")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Arama raporu kaydedildi: {filepath}")
        return filepath

def create_test_query():
    """Test için sorgu görseli oluştur"""
    test_dir = TEST_IMAGES_DIR
    os.makedirs(test_dir, exist_ok=True)
    
    # Basit test görseli
    test_img = Image.new('RGB', IMAGE_SIZE, (128, 128, 128))
    test_path = os.path.join(test_dir, 'test_query.jpg')
    test_img.save(test_path)
    
    return test_path

def main():
    """Ana test fonksiyonu"""
    print("🔍 İç Mekan Benzerlik Arama Sistemi")
    print("=" * 50)
    
    try:
        # Benzerlik arama sistemi başlat
        searcher = SimilaritySearcher()
        
        # Test görseli oluştur
        test_query_path = create_test_query()
        print(f"🎨 Test görseli oluşturuldu: {test_query_path}")
        
        # Benzerlik araması yap
        print("\n🔍 Test araması yapılıyor...")
        indices, scores, message = searcher.search_similar_images(
            test_query_path, 
            k=8,
            similarity_method='cosine'
        )
        
        if len(indices) > 0:
            print(f"✅ {message}")
            
            # Sonuçları görselleştir
            searcher.visualize_search_results(test_query_path, indices, scores)
            
            # Detaylı rapor oluştur
            report = searcher.generate_search_report(test_query_path, indices, scores)
            searcher.save_search_report(report)
            
            # Özet istatistikler
            print(f"\n📊 Arama Sonuçları:")
            print(f"  🎯 Bulunan görsel sayısı: {len(indices)}")
            print(f"  📈 Ortalama benzerlik: {np.mean(scores)*100:.1f}%")
            print(f"  🏆 En yüksek benzerlik: {np.max(scores)*100:.1f}%")
            
        else:
            print(f"❌ {message}")
        
        print("\n🚀 Sonraki adım: Web arayüzü")
        print("   python scripts/web_api.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)