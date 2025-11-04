import numpy as np
import pandas as pd
from datetime import datetime
import json

def validate_input_data(data):
    """Kiểm tra tính hợp lệ của dữ liệu đầu vào"""
    if not data:
        return False, "Dữ liệu trống"
    
    if len(data) < 5:
        return False, "Dữ liệu quá ngắn (cần ít nhất 5 số)"
    
    if not all(isinstance(x, (int, float)) for x in data):
        return False, "Dữ liệu chứa giá trị không phải số"
    
    if not all(3 <= x <= 18 for x in data):
        return False, "Dữ liệu chứa số ngoài phạm vi 3-18"
    
    return True, "Dữ liệu hợp lệ"

def calculate_statistics(data):
    """Tính toán các thống kê cơ bản"""
    if not data:
        return {}
    
    data_array = np.array(data)
    
    return {
        'mean': np.mean(data_array),
        'median': np.median(data_array),
        'std': np.std(data_array),
        'min': np.min(data_array),
        'max': np.max(data_array),
        'tai_count': np.sum(data_array > 10.5),
        'xiu_count': np.sum(data_array <= 10.5),
        'tai_ratio': np.sum(data_array > 10.5) / len(data_array)
    }

def format_prediction_result(prediction, actual=None):
    """Định dạng kết quả dự đoán"""
    trend = "TÀI" if prediction > 10.5 else "XỈU"
    confidence = min(95, abs(prediction - 10.5) / 3.0 * 100)
    
    result = {
        'prediction': float(prediction),
        'trend': trend,
        'confidence': float(confidence),
        'timestamp': datetime.now().isoformat()
    }
    
    if actual is not None:
        result['actual'] = int(actual)
        result['correct'] = (prediction > 10.5) == (actual > 10.5)
    
    return result

def save_prediction_log(prediction_data, filename="prediction_log.json"):
    """Lưu log dự đoán"""
    try:
        # Load log hiện tại
        try:
            with open(filename, 'r') as f:
                log = json.load(f)
        except FileNotFoundError:
            log = []
        
        # Thêm dữ liệu mới
        log.append(prediction_data)
        
        # Giữ log không quá 1000 bản ghi
        if len(log) > 1000:
            log = log[-1000:]
        
        # Lưu log
        with open(filename, 'w') as f:
            json.dump(log, f, indent=2)
            
    except Exception as e:
        print(f"❌ Lỗi khi lưu log: {e}")

def load_prediction_log(filename="prediction_log.json"):
    """Load log dự đoán"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def calculate_rolling_accuracy(log_data, window=50):
    """Tính độ chính xác trượt"""
    if len(log_data) < window:
        return 0.0
    
    recent_data = log_data[-window:]
    correct_predictions = sum(1 for entry in recent_data if entry.get('correct', False))
    
    return correct_predictions / len(recent_data)

def generate_report():
    """Tạo báo cáo tổng hợp"""
    log_data = load_prediction_log()
    
    if not log_data:
        return "Chưa có dữ liệu để tạo báo cáo"
    
    total_predictions = len(log_data)
    correct_predictions = sum(1 for entry in log_data if entry.get('correct', False))
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    report = f"""
BÁO CÁO TỔNG HỢP HỆ THỐNG
========================

Tổng số dự đoán: {total_predictions}
Số dự đoán đúng: {correct_predictions}
Độ chính xác tổng: {accuracy:.1%}

Độ chính xác gần đây (50 lượt): {calculate_rolling_accuracy(log_data):.1%}

Phân phối xu hướng:
- Tài: {sum(1 for entry in log_data if entry.get('trend') == 'TÀI')}
- Xỉu: {sum(1 for entry in log_data if entry.get('trend') == 'XỈU')}

Thời gian chạy: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    return report