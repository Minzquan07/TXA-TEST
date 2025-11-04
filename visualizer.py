import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from fileInteractions import load_actual_results, exportConfig

plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

def plot_tai_xiu_distribution(actual_results):
    """Vẽ biểu đồ phân phối Tài/Xỉu"""
    if not actual_results:
        print("❌ Không có dữ liệu để vẽ biểu đồ")
        return
    
    df = pd.DataFrame(actual_results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Biểu đồ phân phối Tài/Xỉu
    tai_count = len(df[df['actual'] > 10.5])
    xiu_count = len(df[df['actual'] <= 10.5])
    
    ax1.pie([tai_count, xiu_count], 
            labels=['Tài', 'Xỉu'], 
            autopct='%1.1f%%', 
            colors=['#ff6b6b', '#4ecdc4'])
    ax1.set_title('PHÂN PHỐI TÀI/XỈU')
    
    # Biểu đồ phân phối giá trị
    value_counts = df['actual'].value_counts().sort_index()
    ax2.bar(value_counts.index, value_counts.values, color='skyblue', alpha=0.7)
    ax2.axvline(x=10.5, color='red', linestyle='--', linewidth=2, label='Ngưỡng Tài/Xỉu')
    ax2.set_xlabel('Giá trị')
    ax2.set_ylabel('Số lần xuất hiện')
    ax2.set_title('PHÂN PHỐI GIÁ TRỊ')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tai_xiu_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_trend_analysis(actual_results):
    """Vẽ biểu đồ phân tích xu hướng"""
    if len(actual_results) < 10:
        print("❌ Không đủ dữ liệu để phân tích xu hướng")
        return
    
    df = pd.DataFrame(actual_results)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Biểu đồ xu hướng theo thời gian
    trends = [1 if x > 10.5 else 0 for x in df['actual']]
    ax1.plot(range(len(trends)), trends, 'o-', alpha=0.7, linewidth=1, markersize=3)
    ax1.set_xlabel('Thứ tự kết quả')
    ax1.set_ylabel('Xu hướng (1=Tài, 0=Xỉu)')
    ax1.set_title('XU HƯỚNG THEO THỜI GIAN')
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Xỉu', 'Tài'])
    ax1.grid(True, alpha=0.3)
    
    # Biểu đồ chuỗi liên tiếp
    streak_lengths = calculate_streak_lengths(df['actual'])
    ax2.bar(range(len(streak_lengths)), streak_lengths, color='orange', alpha=0.7)
    ax2.set_xlabel('Thứ tự chuỗi')
    ax2.set_ylabel('Độ dài chuỗi')
    ax2.set_title('ĐỘ DÀI CHUỖI TÀI/XỈU LIÊN TIẾP')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trend_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_streak_lengths(data):
    """Tính độ dài các chuỗi liên tiếp"""
    streaks = []
    current_streak = 1
    current_trend = data.iloc[0] > 10.5
    
    for i in range(1, len(data)):
        trend = data.iloc[i] > 10.5
        if trend == current_trend:
            current_streak += 1
        else:
            streaks.append(current_streak)
            current_streak = 1
            current_trend = trend
    
    streaks.append(current_streak)
    return streaks

def plot_model_performance():
    """Vẽ biểu đồ hiệu suất các model"""
    try:
        from helpers import load_prediction_log
        log_data = load_prediction_log()
        
        if not log_data:
            print("❌ Không có log dự đoán")
            return
        
        df = pd.DataFrame(log_data)
        
        # Tính độ chính xác theo model (nếu có thông tin)
        if 'model' in df.columns:
            model_accuracy = df.groupby('model')['correct'].mean().sort_values(ascending=False)
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(model_accuracy.index, model_accuracy.values * 100, 
                          color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57'])
            
            plt.xlabel('Model')
            plt.ylabel('Độ chính xác (%)')
            plt.title('ĐỘ CHÍNH XÁC CÁC MODEL')
            plt.grid(True, alpha=0.3)
            
            # Thêm giá trị trên các cột
            for bar, accuracy in zip(bars, model_accuracy.values * 100):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{accuracy:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
    except Exception as e:
        print(f"❌ Lỗi khi vẽ biểu đồ hiệu suất: {e}")

def plot_confidence_analysis(actual_results):
    """Vẽ biểu đồ phân tích độ tin cậy"""
    if not actual_results:
        return
    
    # Lọc các kết quả có thông tin dự đoán
    predictions_data = []
    for result in actual_results:
        if 'predictions' in result and result['predictions']:
            avg_pred = np.mean(list(result['predictions'].values()))
            confidence = abs(avg_pred - 10.5) / 3.0 * 100
            correct = (avg_pred > 10.5) == (result['actual'] > 10.5)
            predictions_data.append({'confidence': confidence, 'correct': correct})
    
    if not predictions_data:
        return
    
    df = pd.DataFrame(predictions_data)
    
    # Phân nhóm theo độ tin cậy
    bins = [0, 30, 50, 70, 100]
    labels = ['Rất thấp', 'Thấp', 'Trung bình', 'Cao']
    df['confidence_group'] = pd.cut(df['confidence'], bins=bins, labels=labels)
    
    accuracy_by_confidence = df.groupby('confidence_group')['correct'].mean()
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(accuracy_by_confidence.index, accuracy_by_confidence.values * 100, 
                  color=['#ff6b6b', '#feca57', '#48c774', '#3298dc'])
    
    plt.xlabel('Nhóm độ tin cậy')
    plt.ylabel('Độ chính xác (%)')
    plt.title('ĐỘ CHÍNH XÁC THEO MỨC ĐỘ TIN CẬY')
    plt.grid(True, alpha=0.3)
    
    # Thêm giá trị trên các cột
    for bar, accuracy in zip(bars, accuracy_by_confidence.values * 100):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{accuracy:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('confidence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Hàm chính để chạy tất cả visualizations
def run_all_visualizations():
    """Chạy tất cả các biểu đồ"""
    actual_results = load_actual_results()
    
    if not actual_results:
        print("❌ Không có dữ liệu để vẽ biểu đồ")
        return
    
    print("🎨 Đang tạo biểu đồ phân tích...")
    
    plot_tai_xiu_distribution(actual_results)
    plot_trend_analysis(actual_results)
    plot_model_performance()
    plot_confidence_analysis(actual_results)
    
    print("✅ Đã tạo tất cả biểu đồ phân tích")

if __name__ == "__main__":
    run_all_visualizations()