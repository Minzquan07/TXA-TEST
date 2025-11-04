from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from fileInteractions import exportConfig, updateConfig, save_actual_result
import pickle
import numpy as np
import os
from pickle import dump, load

# Tạo thư mục models nếu chưa tồn tại
if not os.path.exists("models"):
    os.makedirs("models")

def train(fileLocation=None):
    """Huấn luyện tất cả các model"""
    if fileLocation is None:
        config = exportConfig()
        fileLocation = config["data_file"]
    
    config = exportConfig()
    input_length = config["input_length"]
    train_data = {"x": [], "y": []}
    
    print("🔄 Đang tải và xử lý dữ liệu...")
    with open(fileLocation, "r+") as file:
        datasets = file.read().splitlines()
        for dataset in datasets:
            if dataset.strip():
                numbers = [int(e) for e in dataset.split(",")]
                for i in range(len(numbers) - input_length - 1):
                    train_data["x"].append(numbers[i : i + input_length])
                    train_data["y"].append(numbers[i + input_length])
    
    print(f"📊 Số lượng mẫu huấn luyện: {len(train_data['x'])}")
    
    # 1. Random Forest
    print("🌲 Đang huấn luyện Random Forest...")
    randomForest = RandomForestRegressor(n_estimators=config["trees_in_the_forest"])
    randomForest.fit(train_data["x"], train_data["y"])
    with open("models/rf.pkl", "wb") as file:
        pickle.dump(randomForest, file)
    print("✅ Đã lưu model Random Forest!")
    
    # 2. SVR
    print("🔍 Đang huấn luyện SVR...")
    svr = SVR(kernel=config["kernel"])
    svr.fit(train_data["x"], train_data["y"])
    with open("models/svr.pkl", "wb") as file:
        pickle.dump(svr, file)
    print("✅ Đã lưu model SVR!")
    
    # 3. Neural Network
    print("🧠 Đang huấn luyện Neural Network...")
    neural = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', max_iter=1000)
    neural.fit(train_data["x"], train_data["y"])
    with open("models/nn.pkl", "wb") as file:
        pickle.dump(neural, file)
    print("✅ Đã lưu model Neural Network!")
    
    # 4. XGBoost
    print("🚀 Đang huấn luyện XGBoost...")
    xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
    xgb.fit(train_data["x"], train_data["y"])
    with open("models/xgb.pkl", "wb") as file:
        pickle.dump(xgb, file)
    print("✅ Đã lưu model XGBoost!")
    
    # 5. Linear Regression
    print("📈 Đang huấn luyện Linear Regression...")
    lr = LinearRegression()
    lr.fit(train_data["x"], train_data["y"])
    with open("models/lr.pkl", "wb") as file:
        pickle.dump(lr, file)
    print("✅ Đã lưu model Linear Regression!")
    
    # 6. Gradient Boosting
    print("📊 Đang huấn luyện Gradient Boosting...")
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=4)
    gb.fit(train_data["x"], train_data["y"])
    with open("models/gb.pkl", "wb") as file:
        pickle.dump(gb, file)
    print("✅ Đã lưu model Gradient Boosting!")

def test(testcase):
    """Kiểm tra độ chính xác trên dữ liệu test"""
    config = exportConfig()
    input_length = config["input_length"]
    test_data = {"x": [], "y": []}

    for i in range(len(testcase) - input_length - 1):
        test_data["x"].append(testcase[i : i + input_length])
        test_data["y"].append(testcase[i + input_length])

    models = {}
    accuracy = {}
    
    # Load models
    model_files = ["rf", "svr", "nn", "xgb", "lr", "gb"]
    for model_name in model_files:
        try:
            with open(f"models/{model_name}.pkl", "rb") as file:
                models[model_name] = pickle.load(file)
        except FileNotFoundError:
            print(f"⚠️  Model {model_name} chưa được huấn luyện")
            continue

    print("🎯 ĐÁNH GIÁ ĐỘ CHÍNH XÁC")
    print("=" * 50)
    
    for model_name, model in models.items():
        predictions = model.predict(test_data["x"])
        accuracy[model_name] = r2_score(test_data["y"], predictions)
        mae_score = mean_absolute_error(test_data["y"], predictions)
        
        print(f"{model_name.upper():>20} | R²: {accuracy[model_name]:.4f} | MAE: {mae_score:.4f}")

    # Tìm model tốt nhất
    if accuracy:
        best_model = max(accuracy, key=accuracy.get)
        print(f"\n🏆 MODEL TỐT NHẤT: {best_model.upper()} (R² = {accuracy[best_model]:.4f})")
    
    return accuracy

def ktest(fileLocation, testcase):
    """Kiểm tra các kernel khác nhau cho SVR"""
    config = exportConfig()
    input_length = config["input_length"]
    test_data = {"x": [], "y": []}
    train_data = {"x": [], "y": []}

    for i in range(len(testcase) - input_length - 1):
        test_data["x"].append(testcase[i : i + input_length])
        test_data["y"].append(testcase[i + input_length])
    
    with open(fileLocation, "r+") as file:
        datasets = file.read().splitlines()
        for dataset in datasets:
            if dataset.strip():
                numbers = [int(e) for e in dataset.split(",")]
                for i in range(len(numbers) - input_length - 1):
                    train_data["x"].append(numbers[i : i + input_length])
                    train_data["y"].append(numbers[i + input_length])

    accuracy = {}
    print("🔍 KIỂM TRA KERNEL SVR")
    print("=" * 30)

    for kernel in ("linear", "poly", "rbf", "sigmoid"):
        svr = SVR(kernel=kernel)
        svr.fit(train_data["x"], train_data["y"])
        prediction = svr.predict(test_data["x"])
        accuracy[kernel] = round(r2_score(test_data["y"], prediction), 4)
        print(f"{kernel:>10} : {accuracy[kernel]:.4f}")

    # Cập nhật kernel tốt nhất vào config
    best_kernel = max(accuracy, key=accuracy.get)
    config = exportConfig()
    config["kernel"] = best_kernel
    updateConfig(config)
    print(f"✅ Đã cập nhật kernel tốt nhất: {best_kernel}")

    return accuracy

def s_predict(inp, model_name):
    """Dự đoán từ model cụ thể"""
    try:
        with open(f"models/{model_name}.pkl", "rb") as file:
            model = pickle.load(file)
            return model.predict([inp])
    except FileNotFoundError:
        print(f"❌ Model {model_name} chưa được huấn luyện")
        return None

def calculate_confidence(predictions, ensemble_result):
    """Tính độ tin cậy dựa trên sự đồng thuận"""
    same_direction = 0
    total_models = len(predictions)
    
    for pred in predictions.values():
        if (pred > 10.5 and ensemble_result > 10.5) or (pred <= 10.5 and ensemble_result <= 10.5):
            same_direction += 1
    
    return (same_direction / total_models) * 100

def weighted_ensemble_predict(inp, weights=None):
    """Dự đoán kết hợp có trọng số"""
    if weights is None:
        weights = {"rf": 0.25, "xgb": 0.25, "gb": 0.2, "nn": 0.15, "svr": 0.1, "lr": 0.05}
    
    predictions = {}
    total_weight = 0
    weighted_sum = 0
    
    for model_name, weight in weights.items():
        pred = s_predict(inp, model_name)
        if pred is not None:
            pred_value = pred[0]
            predictions[model_name] = pred_value
            weighted_sum += pred_value * weight
            total_weight += weight
    
    if total_weight == 0:
        return None, {}
    
    return weighted_sum / total_weight, predictions

def enhanced_predict(inp, ml):
    """Hàm dự đoán nâng cao - tương thích với hàm predict gốc"""
    inp = [int(e) for e in inp] if isinstance(inp, list) else [int(inp)]
    
    if ml == "all":
        res = {}
        for model_name in ["svr", "rf", "nn", "xgb", "lr", "gb"]:
            pred = s_predict(inp, model_name)
            if pred is not None:
                res[model_name] = pred[0]
        
        if not res:
            print("❌ Không có model nào khả dụng. Hãy chạy train trước!")
            return False
            
        print("Kết quả dự đoán:")
        for model_name, pred in res.items():
            trend = "Tài" if pred > 10.5 else "Xỉu"
            print(f"{model_name} : {pred:.4f} -> {trend}")
        
        # Hiển thị dòng để nhập kết quả thực tế
        print("kết quả đúng: ", end="", flush=True)
        
        return {"input": inp, "predictions": res}
        
    elif ml == "ensemble":
        result, all_preds = weighted_ensemble_predict(inp)
        if result is None:
            print("❌ Không có model nào khả dụng. Hãy chạy train trước!")
            return False
            
        print("\n🎲 KẾT QUẢ DỰ ĐOÁN TỪ TẤT CẢ MODELS")
        print("=" * 45)
        for model_name, pred in all_preds.items():
            trend = "Tài" if pred > 10.5 else "Xỉu"
            print(f"{model_name.upper():>15} : {pred:7.4f} -> {trend}")
        
        print("=" * 45)
        print(f"🎯 KẾT QUẢ TỔNG HỢP: {result:.4f}")
        
        # Phân tích kết quả
        confidence = calculate_confidence(all_preds, result)
        final_trend = "Tài" if result > 10.5 else "Xỉu"
        
        print(f"📈 XU HƯỚNG CHUNG: {final_trend}")
        print(f"✅ ĐỘ TIN CẬY: {confidence:.1f}%")
        
        # Khuyến nghị
        print(f"\n💡 KHUYẾN NGHỊ:")
        if 10.25 < result < 10.75:
            print("⚠️  BỎ LƯỢT NÀY (vùng không chắc chắn)")
        elif result > 10.75:
            print("🎯 NÊN CHỌN TÀI")
        else:
            print("🎯 NÊN CHỌN XỈU")
        
        # Hiển thị dòng để nhập kết quả thực tế
        print("\nkết quả đúng: ", end="", flush=True)
        
        return {"input": inp, "predictions": all_preds, "ensemble": result}
        
    elif ml in ["svr", "rf", "nn", "xgb", "lr", "gb"]:
        res = s_predict(inp, ml)
        if res is None:
            print(f"❌ Model {ml} chưa được huấn luyện. Hãy chạy train trước!")
            return False
            
        print(f"\n[{ml.upper()}] KẾT QUẢ DỰ ĐOÁN: {res[0]:.4f}")
        
        # Giữ nguyên logic khuyến nghị gốc
        if 10.25 < res[0] < 10.75:
            print("⚠️  Bạn nên bỏ lượt này")
        elif res[0] > 10.75:
            print("🎯 Khả năng ra Tài là cao hơn")
        else:
            print("🎯 Khả năng ra Xỉu là cao hơn")
        
        # Hiển thị dòng để nhập kết quả thực tế
        print("kết quả đúng: ", end="", flush=True)
        
        return {"input": inp, "predictions": {ml: res[0]}}
    else:
        print("❌ Thuật toán không tồn tại hoặc không được hỗ trợ")
        print("📋 Các lựa chọn: svr, rf, nn, xgb, lr, gb, ensemble, all")
        return False

def predict(inp, ml):
    """Hàm predict gốc - gọi enhanced_predict để có thêm tính năng"""
    return enhanced_predict(inp, ml)