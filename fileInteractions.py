import json
import os
from pickle import dump, load
import pandas as pd

def exportConfig():
    """Xuất cấu hình từ file config.pkl"""
    if not os.path.exists('config.pkl'):
        with open('config.pkl', 'wb') as file:
            config = {
                "kernel": "poly",
                "input_length": 8,
                "trees_in_the_forest": 100,
                "data_file": "DATA/sunwin.txt"
            }
            dump(config, file)
            print("✅ Đã tạo config.pkl mặc định")
    
    with open('config.pkl', 'rb') as file:
        return load(file)

def updateConfig(new_config):
    """Cập nhật cấu hình"""
    with open("config.pkl", "wb") as f:
        dump(new_config, f)

def save_data(data, filename=None):
    """Lưu dữ liệu mới vào file"""
    if filename is None:
        config = exportConfig()
        filename = config["data_file"]
    
    # Đảm bảo thư mục tồn tại
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, "a") as f:
        f.write(",".join(map(str, data)) + "\n")

def save_actual_result(input_data, actual_result):
    """Lưu kết quả thực tế vào file"""
    try:
        # Load kết quả hiện có
        try:
            with open("actual_results.pkl", "rb") as f:
                actual_results = load(f)
        except FileNotFoundError:
            actual_results = []
        
        # Thêm kết quả mới
        actual_results.append({
            "input": input_data,
            "actual": int(actual_result)
        })
        
        # Lưu lại
        with open("actual_results.pkl", "wb") as f:
            dump(actual_results, f)
            
    except Exception as e:
        print(f"❌ Lỗi khi lưu kết quả thực tế: {e}")

def load_actual_results():
    """Load tất cả kết quả thực tế đã lưu"""
    try:
        with open("actual_results.pkl", "rb") as f:
            return load(f)
    except FileNotFoundError:
        return []

def load_data_file(filename=None):
    """Load dữ liệu từ file, tạo file nếu chưa tồn tại"""
    if filename is None:
        config = exportConfig()
        filename = config["data_file"]
    
    # Đảm bảo thư mục tồn tại
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Tạo file nếu chưa tồn tại
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            # Thêm dữ liệu mẫu
            pass
        print(f"✅ Đã tạo file dữ liệu: {filename}")
    
    return filename

def load_data_file(filename=None):
    """Load dữ liệu từ file, tạo file nếu chưa tồn tại"""
    if filename is None:
        config = exportConfig()
        filename = config["data_file"]
    
    # Đảm bảo thư mục tồn tại
    if os.path.dirname(filename):  # Chỉ tạo thư mục nếu có path
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Tạo file nếu chưa tồn tại
    if not os.path.exists(filename):
        with open(filename, 'w', encoding='utf-8') as f:
            # Thêm dữ liệu mẫu cơ bản
            sample_data = [
                [11, 16, 5, 7, 11, 10, 9, 12],
                [10, 10, 13, 11, 14, 9, 15, 6],
                [12, 16, 10, 10, 7, 8, 9, 12],
                [12, 7, 8, 12, 7, 9, 9, 14]
            ]
            for seq in sample_data:
                f.write(",".join(map(str, seq)) + "\n")
        print(f"✅ Đã tạo file dữ liệu: {filename}")
    
    return filename

def get_analysis_stats():
    """Lấy thống kê nhanh về kết quả thực tế"""
    actual_results = load_actual_results()
    
    if not actual_results:
        return {"total": 0, "tai_count": 0, "xiu_count": 0}
    
    actual_values = [result['actual'] for result in actual_results]
    tai_count = sum(1 for x in actual_values if x > 10.5)
    xiu_count = len(actual_values) - tai_count
    total = len(actual_values)
    
    return {
        "total": total,
        "tai_count": tai_count,
        "xiu_count": xiu_count,
        "tai_percentage": tai_count/total*100 if total > 0 else 0,
        "xiu_percentage": xiu_count/total*100 if total > 0 else 0
    }