# scripts/web_api_fixed.py
"""
İç Mekan Benzerlik Arama Sistemi - Web API (Düzeltildi)
Flask tabanlı web arayüzü ve RESTful API
"""

import os
import sys
import base64
import io
import json
from datetime import datetime
from PIL import Image
from flask import Flask, request, jsonify, render_template_string, send_file
from werkzeug.utils import secure_filename
import numpy as np

# Ana dizini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

# Flask uygulaması
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE_MB * 1024 * 1024  # MB to bytes
app.config['UPLOAD_FOLDER'] = UPLOADS_DIR

# Global değişkenler
searcher = None
search_stats = {
    'total_searches': 0,
    'successful_searches': 0,
    'failed_searches': 0,
    'average_response_time': 0.0
}

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🏠 İç Mekan Benzerlik Arama</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .upload-section {
            padding: 40px;
            text-align: center;
            border-bottom: 1px solid #eee;
        }
        
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            margin: 20px 0;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .upload-area:hover {
            border-color: #764ba2;
            background: #f8f9ff;
        }
        
        .upload-area.dragover {
            border-color: #764ba2;
            background: #f0f7ff;
            transform: scale(1.02);
        }
        
        .file-input {
            display: none;
        }
        
        .upload-icon {
            font-size: 3em;
            margin-bottom: 20px;
            color: #667eea;
        }
        
        .controls {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        
        .control-group {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 5px;
        }
        
        .control-group label {
            font-weight: 600;
            color: #555;
        }
        
        .control-group input, .control-group select {
            padding: 8px 15px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 14px;
        }
        
        .search-btn {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 50px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .search-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }
        
        .search-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .results-section {
            padding: 40px;
            display: none;
        }
        
        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .result-item {
            border: 1px solid #ddd;
            border-radius: 10px;
            overflow: hidden;
            transition: all 0.3s ease;
            background: white;
        }
        
        .result-item:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.1);
        }
        
        .result-item img {
            width: 100%;
            height: 150px;
            object-fit: cover;
        }
        
        .result-info {
            padding: 15px;
            text-align: center;
        }
        
        .similarity-score {
            font-weight: 600;
            color: #667eea;
            font-size: 1.1em;
        }
        
        .category-tag {
            background: #f0f7ff;
            color: #667eea;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            margin-top: 5px;
            display: inline-block;
        }
        
        .error {
            background: #ffe6e6;
            color: #d00;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            text-align: center;
            display: none;
        }
        
        .stats {
            background: #f8f9ff;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        
        .stat-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #eee;
        }
        
        .stat-value {
            font-size: 1.5em;
            font-weight: 600;
            color: #667eea;
        }
        
        .stat-label {
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }
        
        .query-preview {
            max-width: 200px;
            margin: 20px auto;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            display: none;
        }
        
        .query-preview img {
            width: 100%;
            height: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏠 İç Mekan Benzerlik Arama</h1>
            <p>Beğendiğiniz iç mekan fotoğrafını yükleyin, benzer tasarımları keşfedin!</p>
        </div>
        
        <div class="upload-section">
            <div class="upload-area" onclick="document.getElementById('fileInput').click()" 
                 ondrop="handleDrop(event)" ondragover="handleDragOver(event)" 
                 ondragleave="handleDragLeave(event)">
                <div class="upload-icon">📷</div>
                <h3>Görsel Yükle</h3>
                <p>Dosyayı buraya sürükleyin veya tıklayarak seçin</p>
                <p style="font-size: 0.9em; color: #666; margin-top: 10px;">
                    Desteklenen formatlar: JPG, PNG, BMP (Max {{ max_file_size }}MB)
                </p>
                <input type="file" id="fileInput" class="file-input" accept="image/*" onchange="handleFileSelect(event)">
            </div>
            
            <div class="query-preview" id="queryPreview">
                <img id="queryImage" src="" alt="Sorgu görseli">
            </div>
            
            <div class="controls">
                <div class="control-group">
                    <label>Sonuç Sayısı</label>
                    <input type="number" id="resultCount" min="1" max="20" value="8">
                </div>
                <div class="control-group">
                    <label>Kategori Filtresi</label>
                    <select id="categoryFilter">
                        <option value="">Tümü</option>
                        <option value="living_rooms">Oturma Odası</option>
                        <option value="bedrooms">Yatak Odası</option>
                        <option value="kitchens">Mutfak</option>
                        <option value="bathrooms">Banyo</option>
                        <option value="dining_rooms">Yemek Odası</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Benzerlik Yöntemi</label>
                    <select id="similarityMethod">
                        <option value="cosine">Kosinüs</option>
                        <option value="euclidean">Öklidyen</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Min. Benzerlik (%)</label>
                    <input type="number" id="minSimilarity" min="0" max="100" value="0" step="5">
                </div>
            </div>
            
            <button class="search-btn" id="searchBtn" onclick="performSearch()" disabled>
                🔍 Benzer Görselleri Ara
            </button>
            
            <div class="error" id="errorMessage"></div>
        </div>
        
        <div class="loading" id="loadingSection">
            <div class="spinner"></div>
            <p>Benzer görseller aranıyor...</p>
            <p style="font-size: 0.9em; color: #666; margin-top: 10px;">
                Bu işlem birkaç saniye sürebilir
            </p>
        </div>
        
        <div class="results-section" id="resultsSection">
            <h2>🎯 Benzer Görseller</h2>
            <div id="searchSummary" style="margin-bottom: 20px; padding: 15px; background: #f0f7ff; border-radius: 8px;"></div>
            <div class="results-grid" id="resultsGrid"></div>
        </div>
        
        <div class="stats" id="statsSection">
            <h3>📊 Sistem İstatistikleri</h3>
            <div class="stats-grid" id="statsGrid"></div>
        </div>
    </div>

    <script>
        let selectedFile = null;
        
        // Sayfa yüklendiğinde istatistikleri getir
        window.onload = function() {
            loadStats();
        };
        
        function loadStats() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        displayStats(data.stats);
                    }
                })
                .catch(error => console.error('İstatistik yüklenirken hata:', error));
        }
        
        function displayStats(stats) {
            const statsGrid = document.getElementById('statsGrid');
            statsGrid.innerHTML = `
                <div class="stat-item">
                    <div class="stat-value">${stats.total_images.toLocaleString()}</div>
                    <div class="stat-label">Toplam Görsel</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${stats.feature_dimension}</div>
                    <div class="stat-label">Özellik Boyutu</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${stats.total_searches}</div>
                    <div class="stat-label">Toplam Arama</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${stats.avg_response_time.toFixed(2)}s</div>
                    <div class="stat-label">Ort. Yanıt Süresi</div>
                </div>
            `;
        }
        
        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                if (validateFile(file)) {
                    selectedFile = file;
                    updateUploadArea(file.name);
                    showQueryPreview(file);
                    document.getElementById('searchBtn').disabled = false;
                }
            }
        }
        
        function validateFile(file) {
            // Dosya boyutu kontrolü
            const maxSize = {{ max_file_size }} * 1024 * 1024; // MB to bytes
            if (file.size > maxSize) {
                showError(`Dosya çok büyük! Maksimum boyut: {{ max_file_size }}MB`);
                return false;
            }
            
            // Dosya tipi kontrolü
            const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp'];
            if (!allowedTypes.includes(file.type)) {
                showError('Desteklenmeyen dosya formatı! JPG, PNG veya BMP kullanın.');
                return false;
            }
            
            return true;
        }
        
        function showQueryPreview(file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const queryPreview = document.getElementById('queryPreview');
                const queryImage = document.getElementById('queryImage');
                queryImage.src = e.target.result;
                queryPreview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
        
        function handleDragOver(event) {
            event.preventDefault();
            event.currentTarget.classList.add('dragover');
        }
        
        function handleDragLeave(event) {
            event.currentTarget.classList.remove('dragover');
        }
        
        function handleDrop(event) {
            event.preventDefault();
            event.currentTarget.classList.remove('dragover');
            
            const files = event.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                if (validateFile(file)) {
                    selectedFile = file;
                    updateUploadArea(file.name);
                    showQueryPreview(file);
                    document.getElementById('searchBtn').disabled = false;
                }
            }
        }
        
        function updateUploadArea(filename) {
            const uploadArea = document.querySelector('.upload-area');
            uploadArea.innerHTML = `
                <div class="upload-icon">✅</div>
                <h3>Seçili Dosya</h3>
                <p>${filename}</p>
                <p style="margin-top: 10px; font-size: 0.9em; color: #666;">
                    Başka bir dosya seçmek için tıklayın
                </p>
            `;
        }
        
        function performSearch() {
            if (!selectedFile) {
                showError('Lütfen bir görsel seçin');
                return;
            }
            
            // UI güncellemeleri
            document.getElementById('loadingSection').style.display = 'block';
            document.getElementById('resultsSection').style.display = 'none';
            document.getElementById('errorMessage').style.display = 'none';
            document.getElementById('searchBtn').disabled = true;
            
            // Form data hazırla
            const formData = new FormData();
            formData.append('image', selectedFile);
            formData.append('k', document.getElementById('resultCount').value);
            formData.append('category_filter', document.getElementById('categoryFilter').value);
            formData.append('similarity_method', document.getElementById('similarityMethod').value);
            formData.append('min_similarity', document.getElementById('minSimilarity').value / 100);
            
            const startTime = Date.now();
            
            // API çağrısı
            fetch('/api/search', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                const responseTime = (Date.now() - startTime) / 1000;
                
                document.getElementById('loadingSection').style.display = 'none';
                document.getElementById('searchBtn').disabled = false;
                
                if (data.success) {
                    displayResults(data.results, data.search_info, responseTime);
                    loadStats(); // İstatistikleri güncelle
                } else {
                    showError(data.error || 'Arama sırasında hata oluştu');
                }
            })
            .catch(error => {
                document.getElementById('loadingSection').style.display = 'none';
                document.getElementById('searchBtn').disabled = false;
                showError('Sunucu hatası: ' + error.message);
            });
        }
        
        function displayResults(results, searchInfo, responseTime) {
            const resultsGrid = document.getElementById('resultsGrid');
            const resultsSection = document.getElementById('resultsSection');
            const searchSummary = document.getElementById('searchSummary');
            
            // Arama özeti
            searchSummary.innerHTML = `
                <strong>🔍 Arama Sonuçları:</strong> ${results.length} görsel bulundu 
                (${responseTime.toFixed(2)} saniye) 
                ${searchInfo.category_filter ? `| Kategori: ${searchInfo.category_filter}` : ''}
                ${searchInfo.min_similarity > 0 ? `| Min. benzerlik: ${(searchInfo.min_similarity * 100).toFixed(0)}%` : ''}
            `;
            
            resultsGrid.innerHTML = '';
            
            results.forEach((result, index) => {
                const resultItem = document.createElement('div');
                resultItem.className = 'result-item';
                
                // Kategori adını düzelt
                const categoryName = result.category.replace('_', ' ').replace(/\\b\\w/g, function(l) { 
                    return l.toUpperCase(); 
                });
                
                resultItem.innerHTML = `
                    <img src="data:image/jpeg;base64,${result.image_data}" alt="Benzer görsel ${index + 1}">
                    <div class="result-info">
                        <div class="similarity-score">${(result.similarity_score * 100).toFixed(1)}%</div>
                        <div class="category-tag">${categoryName}</div>
                        <div style="margin-top: 8px; font-size: 0.8em; color: #666;">
                            Küme: ${result.style_cluster} | ${result.filename}
                        </div>
                    </div>
                `;
                resultsGrid.appendChild(resultItem);
            });
            
            resultsSection.style.display = 'block';
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        }
        
        function showError(message) {
            const errorElement = document.getElementById('errorMessage');
            errorElement.textContent = message;
            errorElement.style.display = 'block';
            
            // 5 saniye sonra gizle
            setTimeout(() => {
                errorElement.style.display = 'none';
            }, 5000);
        }
    </script>
</body>
</html>
"""

def initialize_searcher():
    """Benzerlik arama sistemini başlat"""
    global searcher
    if searcher is None:
        print("🔄 Benzerlik arama sistemi başlatılıyor...")
        try:
            from similarity_search_fixed import SimilaritySearcher  # Düzeltilmiş dosyayı kullan
            searcher = SimilaritySearcher()
            print("✅ Benzerlik arama sistemi hazır")
        except Exception as e:
            print(f"❌ Benzerlik arama sistemi başlatılamadı: {e}")
            raise

def allowed_file(filename):
    """Dosya uzantısı kontrolü"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ['jpg', 'jpeg', 'png', 'bmp']

@app.route('/')
def index():
    """Ana sayfa"""
    return render_template_string(HTML_TEMPLATE, max_file_size=MAX_FILE_SIZE_MB)

@app.route('/api/search', methods=['POST'])
def api_search():
    """Benzerlik arama API endpoint'i"""
    global search_stats
    import time
    start_time = time.time()
    
    try:
        search_stats['total_searches'] += 1
        
        # Dosya kontrolü
        if 'image' not in request.files:
            search_stats['failed_searches'] += 1
            return jsonify({'success': False, 'error': 'Görsel bulunamadı'}), 400
        
        file = request.files['image']
        if file.filename == '':
            search_stats['failed_searches'] += 1
            return jsonify({'success': False, 'error': 'Dosya seçilmedi'}), 400
        
        if not allowed_file(file.filename):
            search_stats['failed_searches'] += 1
            return jsonify({'success': False, 'error': 'Desteklenmeyen dosya formatı'}), 400
        
        # Parametreler
        k = int(request.form.get('k', 5))
        category_filter = request.form.get('category_filter', None)
        if category_filter == '':
            category_filter = None
        similarity_method = request.form.get('similarity_method', 'cosine')
        min_similarity = float(request.form.get('min_similarity', 0.0))
        
        # Dosyayı geçici olarak kaydet
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{int(time.time())}_{filename}")
        file.save(temp_path)
        
        try:
            # Arama yap
            indices, scores, message = searcher.search_similar_images(
                temp_path, 
                k=k,
                category_filter=[category_filter] if category_filter else None,
                similarity_method=similarity_method,
                min_similarity=min_similarity
            )
            
            if len(indices) == 0:
                search_stats['failed_searches'] += 1
                return jsonify({
                    'success': False, 
                    'error': message or 'Benzer görsel bulunamadı'
                })
            
            # Sonuçları hazırla
            results = []
            for idx, score in zip(indices, scores):
                result_path = searcher.metadata.iloc[idx]['filepath']
                
                # Görseli base64'e çevir
                try:
                    with open(result_path, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode()
                    
                    results.append({
                        'image_data': img_data,
                        'similarity_score': float(score),
                        'filename': searcher.metadata.iloc[idx]['filename'],
                        'category': searcher.metadata.iloc[idx]['category'],
                        'style_cluster': int(searcher.style_clusters[idx]),
                        'file_path': result_path
                    })
                except Exception as e:
                    print(f"Görsel yüklenirken hata: {e}")
                    continue
            
            # Başarı istatistikleri güncelle
            search_stats['successful_searches'] += 1
            response_time = time.time() - start_time
            
            # Ortalama yanıt süresini güncelle
            total_successful = search_stats['successful_searches']
            current_avg = search_stats['average_response_time']
            search_stats['average_response_time'] = (current_avg * (total_successful - 1) + response_time) / total_successful
            
            return jsonify({
                'success': True,
                'results': results,
                'search_info': {
                    'query_filename': filename,
                    'total_found': len(results),
                    'response_time': response_time,
                    'category_filter': category_filter,
                    'similarity_method': similarity_method,
                    'min_similarity': min_similarity
                }
            })
            
        finally:
            # Geçici dosyayı sil
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        search_stats['failed_searches'] += 1
        print(f"API Hatası: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Sunucu hatası: {str(e)}'}), 500

@app.route('/api/stats')
def api_stats():
    """Sistem istatistikleri"""
    try:
        if searcher is None:
            return jsonify({'success': False, 'error': 'Sistem henüz hazır değil'})
        
        stats = {
            'total_images': len(searcher.features),
            'feature_dimension': searcher.features.shape[1],
            'categories': len(CATEGORIES),
            'total_searches': search_stats['total_searches'],
            'successful_searches': search_stats['successful_searches'],
            'failed_searches': search_stats['failed_searches'],
            'success_rate': (search_stats['successful_searches'] / max(1, search_stats['total_searches'])) * 100,
            'avg_response_time': search_stats['average_response_time']
        }
        
        return jsonify({'success': True, 'stats': stats})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/categories')
def api_categories():
    """Mevcut kategoriler"""
    try:
        if searcher is None:
            return jsonify({'success': False, 'error': 'Sistem henüz hazır değil'})
        
        category_stats = searcher.metadata['category'].value_counts().to_dict()
        
        return jsonify({
            'success': True,
            'categories': category_stats,
            'total_categories': len(category_stats)
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/health')
def api_health():
    """Sistem sağlık durumu"""
    try:
        health_status = {
            'status': 'healthy' if searcher is not None else 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'features_loaded': searcher is not None and searcher.features is not None,
                'metadata_loaded': searcher is not None and searcher.metadata is not None,
                'models_loaded': searcher is not None and searcher.pca_model is not None
            }
        }
        
        return jsonify(health_status)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        })

@app.errorhandler(413)
def too_large(e):
    """Dosya çok büyük hatası"""
    return jsonify({
        'success': False, 
        'error': f'Dosya çok büyük! Maksimum boyut: {MAX_FILE_SIZE_MB}MB'
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Sayfa bulunamadı"""
    return jsonify({'success': False, 'error': 'Endpoint bulunamadı'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Sunucu hatası"""
    return jsonify({'success': False, 'error': 'Sunucu hatası'}), 500

def main():
    """Ana fonksiyon"""
    print("🌐 İç Mekan Benzerlik Web API'si")
    print("=" * 50)
    
    # Upload klasörünü oluştur
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    
    try:
        # Benzerlik arama sistemini başlat
        initialize_searcher()
        
        print(f"\n🚀 Web sunucusu başlatılıyor...")
        print(f"📍 Adres: http://{FLASK_HOST}:{FLASK_PORT}")
        print(f"🔧 Debug modu: {'Açık' if FLASK_DEBUG else 'Kapalı'}")
        print(f"📊 Maksimum dosya boyutu: {MAX_FILE_SIZE_MB}MB")
        print(f"📁 Upload klasörü: {UPLOADS_DIR}")
        print(f"\n⏹️  Durdurmak için Ctrl+C")
        print("=" * 50)
        
        # Flask uygulamasını çalıştır
        app.run(
            host=FLASK_HOST,
            port=FLASK_PORT,
            debug=FLASK_DEBUG,
            threaded=True
        )
        
    except KeyboardInterrupt:
        print("\n⏹️ Web sunucusu durduruldu")
    except Exception as e:
        print(f"\n❌ Web sunucusu başlatılamadı: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()