from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from fileInteractions import exportConfig, save_actual_result, load_data_file
import pickle
import numpy as np
import os
from pickle import dump, load
import warnings
warnings.filterwarnings('ignore')

# Tạo thư mục models nếu chưa tồn tại
if not os.path.exists("models"):
    os.makedirs("models")

class EnhancedPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def create_advanced_models(self):
        """Tạo các model nâng cao với hyperparameter tuning"""
        models = {
            'rf_enhanced': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42
            ),
            'xgb_enhanced': XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'lgbm': LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=15,
                min_data_in_leaf=1,
                random_state=42
            ),
            'gbm_enhanced': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'svr_enhanced': SVR(
                kernel='rbf',
                C=1.0,
                gamma='scale'
            ),
            'mlp_enhanced': MLPRegressor(
                hidden_layer_sizes=(50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='constant',
                max_iter=1000,
                early_stopping=False,
                random_state=42
            ),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1)
        }
        return models

    def prepare_features(self, data):
        """Chuẩn bị features nâng cao"""
        X = []
        y = []
        
        for sequence in data:
            if len(sequence) < 8:
                continue
                
            # Features cơ bản
            recent = sequence[-8:]
            
            # Features thống kê
            mean_val = np.mean(recent)
            std_val = np.std(recent)
            median_val = np.median(recent)
            
            # Features xu hướng
            if len(recent) > 1:
                trend = np.polyfit(range(len(recent)), recent, 1)[0]
            else:
                trend = 0
            
            # Features chu kỳ
            if len(recent) > 1:
                diff_features = [
                    recent[i] - recent[i-1] for i in range(1, len(recent))
                ]
                avg_diff = np.mean(diff_features)
            else:
                avg_diff = 0
            
            # Features Tài/Xỉu
            tai_count = sum(1 for x in recent if x > 10.5)
            xiu_count = len(recent) - tai_count
            
            # Kết hợp tất cả features
            features = recent + [
                mean_val, std_val, median_val, trend, avg_diff, 
                tai_count, xiu_count
            ]
            
            X.append(features)
            y.append(sequence[-1])
        
        return np.array(X), np.array(y)

    def train_enhanced(self, fileLocation=None):
        """Huấn luyện model nâng cao"""
        fileLocation = load_data_file(fileLocation)
        config = exportConfig()
        
        print("🔄 Đang chuẩn bị dữ liệu nâng cao...")
        
        # Load và xử lý dữ liệu
        all_sequences = []
        with open(fileLocation, "r+", encoding='utf-8') as file:
            datasets = file.read().splitlines()
            for set_line in datasets:
                if set_line.strip():
                    set_line = set_line.replace('\t', ',').replace(' ', ',')
                    numbers = [int(e.strip()) for e in set_line.split(',') if e.strip()]
                    if len(numbers) >= 8:
                        all_sequences.append(numbers)
        
        # Thêm dữ liệu từ kết quả thực tế
        try:
            with open("actual_results.pkl", "rb") as f:
                actual_results = load(f)
                for result in actual_results:
                    if len(result["input"]) >= 5:
                        sequence = result["input"] + [result["actual"]]
                        if len(sequence) >= 8:
                            all_sequences.append(sequence)
        except FileNotFoundError:
            pass
        
        if len(all_sequences) == 0:
            print("❌ Không đủ dữ liệu để huấn luyện!")
            return False
        
        # Chuẩn bị features
        X, y = self.prepare_features(all_sequences)
        
        if len(X) == 0:
            print("❌ Không tạo được features từ dữ liệu!")
            return False
            
        print(f"📊 Số lượng mẫu huấn luyện: {len(X)}")
        print(f"🔢 Số features: {X.shape[1]}")
        
        # Kiểm tra số lượng mẫu
        if len(X) < 5:
            print("⚠️  CẢNH BÁO: Dữ liệu quá ít, chỉ có thể huấn luyện cơ bản")
            return self.train_basic_models(X, y)
        
        # Chuẩn hóa dữ liệu
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        
        # Huấn luyện các model
        models = self.create_advanced_models()
        
        print("🚀 Đang huấn luyện các model nâng cao...")
        
        best_score = -float('inf')
        best_model_name = None
        
        for name, model in models.items():
            try:
                print(f"🔧 Đang huấn luyện {name}...")
                
                if name in ['svr_enhanced', 'mlp_enhanced', 'ridge', 'lasso']:
                    model.fit(X_scaled, y)
                else:
                    model.fit(X, y)
                
                # Đánh giá với cross-validation (chỉ khi đủ dữ liệu)
                if len(X) >= 10:
                    if name not in ['svr_enhanced', 'mlp_enhanced']:
                        cv_splits = min(5, len(X))
                        scores = cross_val_score(model, X, y, cv=cv_splits, scoring='r2')
                        avg_score = np.mean(scores)
                    else:
                        cv_splits = min(5, len(X))
                        scores = cross_val_score(model, X_scaled, y, cv=cv_splits, scoring='r2')
                        avg_score = np.mean(scores)
                    
                    print(f"   ✅ {name}: R² = {avg_score:.4f}")
                else:
                    # Dự đoán trên chính training data để có đánh giá sơ bộ
                    if name in ['svr_enhanced', 'mlp_enhanced', 'ridge', 'lasso']:
                        pred = model.predict(X_scaled)
                    else:
                        pred = model.predict(X)
                    avg_score = r2_score(y, pred)
                    print(f"   ✅ {name}: R² (train) = {avg_score:.4f}")
                
                # Lưu model
                with open(f"models/{name}.pkl", "wb") as f:
                    pickle.dump(model, f)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model_name = name
                    
            except Exception as e:
                print(f"   ❌ Lỗi với {name}: {e}")
        
        if best_model_name:
            print(f"\n🏆 MODEL TỐT NHẤT: {best_model_name} (R² = {best_score:.4f})")
        else:
            print(f"\n❌ Không có model nào huấn luyện thành công")
            return False
        
        # Lưu scaler
        with open("models/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
            
        return True

    def train_basic_models(self, X, y):
        """Huấn luyện models cơ bản khi dữ liệu ít"""
        print("🔧 Chuyển sang huấn luyện models cơ bản (dữ liệu ít)...")
        
        models = {
            'rf_basic': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
            'xgb_basic': XGBRegressor(n_estimators=50, max_depth=3, random_state=42),
            'lr_basic': LinearRegression()
        }
        
        best_score = -float('inf')
        best_model_name = None
        
        for name, model in models.items():
            try:
                model.fit(X, y)
                pred = model.predict(X)
                score = r2_score(y, pred)
                
                print(f"   ✅ {name}: R² (train) = {score:.4f}")
                
                with open(f"models/{name}.pkl", "wb") as f:
                    pickle.dump(model, f)
                
                if score > best_score:
                    best_score = score
                    best_model_name = name
                    
            except Exception as e:
                print(f"   ❌ Lỗi với {name}: {e}")
        
        if best_model_name:
            print(f"\n🏆 MODEL TỐT NHẤT: {best_model_name} (R² = {best_score:.4f})")
            return True
        else:
            print(f"\n❌ Không thể huấn luyện model nào")
            return False

    def predict_enhanced(self, input_sequence):
        """Dự đoán nâng cao với features phong phú"""
        if not self.is_fitted:
            try:
                with open("models/scaler.pkl", "rb") as f:
                    self.scaler = pickle.load(f)
                self.is_fitted = True
            except FileNotFoundError:
                print("❌ Chưa có model được huấn luyện!")
                return None, {}
        
        # Chuẩn bị features cho input
        recent = input_sequence[-8:]
        
        # Tính các features
        mean_val = np.mean(recent)
        std_val = np.std(recent)
        median_val = np.median(recent)
        
        if len(recent) > 1:
            trend = np.polyfit(range(len(recent)), recent, 1)[0]
        else:
            trend = 0
            
        if len(recent) > 1:
            diff_features = [recent[i] - recent[i-1] for i in range(1, len(recent))]
            avg_diff = np.mean(diff_features)
        else:
            avg_diff = 0
        
        tai_count = sum(1 for x in recent if x > 10.5)
        xiu_count = len(recent) - tai_count
        
        features = recent + [
            mean_val, std_val, median_val, trend, avg_diff, 
            tai_count, xiu_count
        ]
        
        features = np.array(features).reshape(1, -1)
        
        # Dự đoán với tất cả models
        predictions = {}
        model_files = ['rf_enhanced', 'xgb_enhanced', 'lgbm', 'gbm_enhanced', 
                      'svr_enhanced', 'mlp_enhanced', 'ridge', 'lasso',
                      'rf_basic', 'xgb_basic', 'lr_basic']
        
        for model_name in model_files:
            try:
                with open(f"models/{model_name}.pkl", "rb") as f:
                    model = pickle.load(f)
                
                if model_name in ['svr_enhanced', 'mlp_enhanced', 'ridge', 'lasso']:
                    features_scaled = self.scaler.transform(features)
                    pred = model.predict(features_scaled)[0]
                else:
                    pred = model.predict(features)[0]
                
                predictions[model_name] = pred
                
            except FileNotFoundError:
                continue
        
        if not predictions:
            return None, {}
        
        # Weighted ensemble
        weights = {
            'xgb_enhanced': 0.2, 'lgbm': 0.15, 'rf_enhanced': 0.15, 'gbm_enhanced': 0.1,
            'mlp_enhanced': 0.08, 'svr_enhanced': 0.05, 'ridge': 0.04, 'lasso': 0.03,
            'xgb_basic': 0.1, 'rf_basic': 0.08, 'lr_basic': 0.02
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for model_name, pred in predictions.items():
            weight = weights.get(model_name, 0.05)
            weighted_sum += pred * weight
            total_weight += weight
        
        final_prediction = weighted_sum / total_weight
        
        return final_prediction, predictions

# Tạo instance toàn cục
enhanced_predictor = EnhancedPredictor()

def train_enhanced_models():
    """Huấn luyện models nâng cao"""
    success = enhanced_predictor.train_enhanced()
    return success

def predict_enhanced(input_sequence, algorithm="enhanced"):
    """Dự đoán với hệ thống nâng cao"""
    if algorithm == "enhanced":
        result, all_preds = enhanced_predictor.predict_enhanced(input_sequence)
        
        if result is None:
            print("❌ Chưa có model nâng cao được huấn luyện!")
            return False
        
        print("\n🚀 KẾT QUẢ DỰ ĐOÁN NÂNG CAO")
        print("=" * 50)
        
        for model_name, pred in all_preds.items():
            trend = "TÀI" if pred > 10.5 else "XỈU"
            confidence = min(95, abs(pred - 10.5) / 3.0 * 100)
            print(f"{model_name:>15} : {pred:7.4f} -> {trend} ({confidence:.1f}%)")
        
        print("=" * 50)
        print(f"🎯 KẾT QUẢ TỔNG HỢP: {result:.4f}")
        
        # Phân tích chi tiết
        final_trend = "TÀI" if result > 10.5 else "XỈU"
        confidence = min(95, abs(result - 10.5) / 3.0 * 100)
        
        print(f"📈 XU HƯỚNG: {final_trend}")
        print(f"✅ ĐỘ TIN CẬY: {confidence:.1f}%")
        
        # Khuyến nghị chiến lược
        if confidence > 70:
            print("💎 KHUYẾN NGHỊ: TIN TƯỞNG - ĐẶT CƯỢC MẠNH")
        elif confidence > 55:
            print("🎯 KHUYẾN NGHỊ: TIN TƯỚNG - ĐẶT CƯỢC VỪA")
        elif confidence > 45:
            print("⚠️  KHUYẾN NGHỊ: THẬN TRỌNG - ĐẶT CƯỢC NHẸ")
        else:
            print("🚫 KHUYẾN NGHỊ: BỎ LƯỢT - QUÁ RỦI RO")
        
        print("\nkết quả đúng: ", end="", flush=True)
        
        return {"input": input_sequence, "predictions": all_preds, "ensemble": result}
    
    else:
        # Fallback về hệ thống cũ
        from mlModels import predict as legacy_predict
        return legacy_predict(input_sequence, algorithm)