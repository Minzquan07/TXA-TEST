import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from datetime import datetime
import pandas as pd
from fileInteractions import load_actual_results, exportConfig
import os
from pickle import load

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

def analyze_actual_results():
    """Phân tích toàn diện kết quả thực tế đã lưu"""
    actual_results = load_actual_results()
    
    if not actual_results:
        print("❌ Chưa có kết quả thực tế nào được lưu!")
        return
    
    print(f"📊 PHÂN TÍCH {len(actual_results)} KẾT QUẢ THỰC TẾ")
    print("=" * 50)
    
    # Chuyển thành DataFrame để phân tích
    df = pd.DataFrame(actual_results)
    
    # 1. Thống kê cơ bản
    print("\n📈 THỐNG KÊ CƠ BẢN:")
    print(f"• Tổng số kết quả: {len(df)}")
    print(f"• Giá trị nhỏ nhất: {df['actual'].min()}")
    print(f"• Giá trị lớn nhất: {df['actual'].max()}")
    print(f"• Trung bình: {df['actual'].mean():.2f}")
    print(f"• Trung vị: {df['actual'].median()}")
    print(f"• Độ lệch chuẩn: {df['actual'].std():.2f}")
    
    # 2. Phân phối Tài/Xỉu
    tai_count = len(df[df['actual'] > 10.5])
    xiu_count = len(df[df['actual'] <= 10.5])
    total = len(df)
    
    print(f"\n🎲 PHÂN PHỐI TÀI/XỈU:")
    print(f"• Tài: {tai_count} ({tai_count/total*100:.1f}%)")
    print(f"• Xỉu: {xiu_count} ({xiu_count/total*100:.1f}%)")
    
    # 3. Phân tích xu hướng
    print(f"\n📊 PHÂN TÍCH XU HƯỚNG:")
    
    # Xu hướng gần đây (10 kết quả cuối)
    recent = df.tail(min(10, len(df)))
    recent_tai = len(recent[recent['actual'] > 10.5])
    recent_xiu = len(recent[recent['actual'] <= 10.5])
    
    print(f"• 10 kết quả gần nhất: Tài {recent_tai}, Xỉu {recent_xiu}")
    
    # Chuỗi liên tiếp
    consecutive = analyze_consecutive_trends(df)
    print(f"• Chuỗi Tài dài nhất: {consecutive['max_tai']}")
    print(f"• Chuỗi Xỉu dài nhất: {consecutive['max_xiu']}")
    
    # 4. Phân tích theo giá trị cụ thể
    value_analysis = analyze_value_distribution(df)
    print(f"\n🔢 PHÂN TÍCH THEO GIÁ TRỊ:")
    print(f"• Số xuất hiện nhiều nhất: {value_analysis['most_common']}")
    print(f"• Tần suất các số:")
    for value, count in value_analysis['value_counts'].most_common(5):
        print(f"  {value}: {count} lần ({count/total*100:.1f}%)")
    
    # 5. Độ chính xác của dự đoán (nếu có thông tin dự đoán)
    if any('predictions' in row for index, row in df.iterrows()):
        accuracy_analysis = analyze_prediction_accuracy(df)
        print(f"\n🎯 ĐỘ CHÍNH XÁC DỰ ĐOÁN:")
        for model, acc in accuracy_analysis.items():
            print(f"• {model.upper()}: {acc['accuracy']:.1f}%")
    
    # 6. Vẽ biểu đồ
    plot_analysis_results(df, actual_results)
    
    return df

def analyze_consecutive_trends(df):
    """Phân tích chuỗi liên tiếp Tài/Xỉu"""
    trends = []
    current_trend = None
    current_length = 0
    max_tai = 0
    max_xiu = 0
    
    for actual in df['actual']:
        trend = 'tai' if actual > 10.5 else 'xiu'
        
        if trend == current_trend:
            current_length += 1
        else:
            if current_trend == 'tai':
                max_tai = max(max_tai, current_length)
            elif current_trend == 'xiu':
                max_xiu = max(max_xiu, current_length)
            
            current_trend = trend
            current_length = 1
    
    # Cập nhật cho chuỗi cuối cùng
    if current_trend == 'tai':
        max_tai = max(max_tai, current_length)
    elif current_trend == 'xiu':
        max_xiu = max(max_xiu, current_length)
    
    return {'max_tai': max_tai, 'max_xiu': max_xiu}

def analyze_value_distribution(df):
    """Phân tích phân phối giá trị"""
    value_counts = Counter(df['actual'])
    most_common = value_counts.most_common(1)[0][0] if value_counts else None
    
    return {
        'value_counts': value_counts,
        'most_common': most_common
    }

def analyze_prediction_accuracy(df):
    """Phân tích độ chính xác của các model"""
    accuracy_results = {}
    
    for index, row in df.iterrows():
        if 'predictions' in row and row['predictions']:
            actual = row['actual']
            predictions = row['predictions']
            
            for model, pred_value in predictions.items():
                if model not in accuracy_results:
                    accuracy_results[model] = {'correct': 0, 'total': 0}
                
                pred_trend = 'tai' if pred_value > 10.5 else 'xiu'
                actual_trend = 'tai' if actual > 10.5 else 'xiu'
                
                accuracy_results[model]['total'] += 1
                if pred_trend == actual_trend:
                    accuracy_results[model]['correct'] += 1
    
    # Tính phần trăm chính xác
    for model, stats in accuracy_results.items():
        if stats['total'] > 0:
            stats['accuracy'] = (stats['correct'] / stats['total']) * 100
        else:
            stats['accuracy'] = 0
    
    return accuracy_results

def plot_analysis_results(df, actual_results):
    """Vẽ biểu đồ phân tích kết quả"""
    if len(df) < 5:
        print("📊 Không đủ dữ liệu để vẽ biểu đồ")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('PHÂN TÍCH KẾT QUẢ THỰC TẾ', fontsize=16, fontweight='bold')
    
    # 1. Biểu đồ phân phối Tài/Xỉu
    tai_xiu_counts = [len(df[df['actual'] > 10.5]), len(df[df['actual'] <= 10.5])]
    axes[0, 0].pie(tai_xiu_counts, labels=['Tài', 'Xỉu'], autopct='%1.1f%%', colors=['#ff6b6b', '#4ecdc4'])
    axes[0, 0].set_title('PHÂN PHỐI TÀI/XỈU')
    
    # 2. Biểu đồ phân phối giá trị
    value_counts = Counter(df['actual'])
    values = list(value_counts.keys())
    counts = list(value_counts.values())
    
    axes[0, 1].bar(values, counts, color='skyblue', alpha=0.7)
    axes[0, 1].set_xlabel('Giá trị')
    axes[0, 1].set_ylabel('Số lần xuất hiện')
    axes[0, 1].set_title('PHÂN PHỐI GIÁ TRỊ')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Biểu đồ xu hướng theo thời gian
    trends = ['Tài' if x > 10.5 else 'Xỉu' for x in df['actual']]
    trend_numeric = [1 if x == 'Tài' else 0 for x in trends]
    
    axes[1, 0].plot(range(len(trend_numeric)), trend_numeric, 'o-', alpha=0.7, linewidth=2)
    axes[1, 0].set_xlabel('Thứ tự kết quả')
    axes[1, 0].set_ylabel('Xu hướng (1=Tài, 0=Xỉu)')
    axes[1, 0].set_title('XU HƯỚNG THEO THỜI GIAN')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_yticklabels(['Xỉu', 'Tài'])
    
    # 4. Biểu đồ độ chính xác model
    if any('predictions' in row for index, row in df.iterrows()):
        accuracy_analysis = analyze_prediction_accuracy(df)
        models = list(accuracy_analysis.keys())
        accuracies = [accuracy_analysis[model]['accuracy'] for model in models]
        
        bars = axes[1, 1].bar(models, accuracies, color=['#ff9ff3', '#f368e0', '#ff6b6b', '#ee5a24', '#00d2d3', '#54a0ff'])
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_ylabel('Độ chính xác (%)')
        axes[1, 1].set_title('ĐỘ CHÍNH XÁC CÁC MODEL')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Thêm giá trị trên mỗi cột
        for bar, accuracy in zip(bars, accuracies):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                           f'{accuracy:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Đã lưu biểu đồ phân tích vào: analysis_results.png")

def analyze_recent_performance(days=7):
    """Phân tích hiệu suất gần đây"""
    actual_results = load_actual_results()
    
    if not actual_results:
        print("❌ Chưa có kết quả thực tế nào được lưu!")
        return
    
    df = pd.DataFrame(actual_results)
    
    # Lấy kết quả gần đây
    recent_df = df.tail(min(days * 10, len(df)))
    
    print(f"\n📈 PHÂN TÍCH HIỆU SUẤT {len(recent_df)} KẾT QUẢ GẦN ĐÂY")
    print("=" * 50)
    
    tai_count = len(recent_df[recent_df['actual'] > 10.5])
    xiu_count = len(recent_df[recent_df['actual'] <= 10.5])
    total = len(recent_df)
    
    print(f"• Tài: {tai_count} ({tai_count/total*100:.1f}%)")
    print(f"• Xỉu: {xiu_count} ({xiu_count/total*100:.1f}%)")
    
    # Phân tích lợi nhuận giả định
    print(f"\n💰 PHÂN TÍCH LỢI NHUẬN GIẢ ĐỊNH:")
    print(f"• Tỷ lệ thắng: {max(tai_count, xiu_count)/total*100:.1f}%")
    print(f"• Lợi nhuận tiềm năng: {(max(tai_count, xiu_count) - min(tai_count, xiu_count)) / total * 100:.1f}%")

def export_analysis_report():
    """Xuất báo cáo phân tích chi tiết"""
    actual_results = load_actual_results()
    
    if not actual_results:
        print("❌ Chưa có kết quả thực tế nào được lưu!")
        return
    
    df = pd.DataFrame(actual_results)
    
    # Tạo báo cáo
    report = f"""
BÁO CÁO PHÂN TÍCH KẾT QUẢ THỰC TẾ
================================

Tổng số kết quả: {len(df)}
Thời gian phân tích: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

THỐNG KÊ CƠ BẢN:
- Giá trị trung bình: {df['actual'].mean():.2f}
- Giá trị trung vị: {df['actual'].median()}
- Độ lệch chuẩn: {df['actual'].std():.2f}
- Min-Max: {df['actual'].min()} - {df['actual'].max()}

PHÂN PHỐI TÀI/XỈU:
- Tài: {len(df[df['actual'] > 10.5])} ({len(df[df['actual'] > 10.5])/len(df)*100:.1f}%)
- Xỉu: {len(df[df['actual'] <= 10.5])} ({len(df[df['actual'] <= 10.5])/len(df)*100:.1f}%)

SỐ XUẤT HIỆN NHIỀU NHẤT:
"""
    
    value_counts = Counter(df['actual'])
    for value, count in value_counts.most_common(10):
        report += f"    - {value}: {count} lần ({count/len(df)*100:.1f}%)\n"
    
    # Lưu báo cáo
    with open('analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("✅ Đã xuất báo cáo phân tích vào: analysis_report.txt")
    return report

def compare_with_historical():
    """So sánh với dữ liệu lịch sử"""
    config = exportConfig()
    historical_file = config["data_file"]
    
    # Load dữ liệu lịch sử
    try:
        with open(historical_file, "r") as file:
            historical_data = []
            for line in file.read().splitlines():
                if line.strip():
                    numbers = [int(e.strip()) for e in line.replace('\t', ',').replace(' ', ',').split(',') if e.strip()]
                    historical_data.extend(numbers)
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file lịch sử: {historical_file}")
        return
    
    # Load kết quả thực tế
    actual_results = load_actual_results()
    if not actual_results:
        print("❌ Chưa có kết quả thực tế nào được lưu!")
        return
    
    actual_values = [result['actual'] for result in actual_results]
    
    print(f"\n📊 SO SÁNH VỚI DỮ LIỆU LỊCH SỬ")
    print("=" * 40)
    print(f"• Dữ liệu lịch sử: {len(historical_data)} số")
    print(f"• Kết quả thực tế: {len(actual_values)} số")
    
    # So sánh phân phối
    hist_tai = len([x for x in historical_data if x > 10.5])
    hist_xiu = len([x for x in historical_data if x <= 10.5])
    actual_tai = len([x for x in actual_values if x > 10.5])
    actual_xiu = len([x for x in actual_values if x <= 10.5])
    
    print(f"\n🎲 SO SÁNH PHÂN PHỐI:")
    print(f"• Lịch sử - Tài: {hist_tai/len(historical_data)*100:.1f}%")
    print(f"• Thực tế - Tài: {actual_tai/len(actual_values)*100:.1f}%")
    print(f"• Lịch sử - Xỉu: {hist_xiu/len(historical_data)*100:.1f}%")
    print(f"• Thực tế - Xỉu: {actual_xiu/len(actual_values)*100:.1f}%")

def analyze_patterns():
    """Phân tích pattern thường gặp trong dữ liệu Sunwin"""
    config = exportConfig()
    data_file = config["data_file"]
    
    print("🎯 Phân tích pattern dữ liệu Sunwin...")
    print("🔍 PHÂN TÍCH PATTERN DỮ LIỆU SUNWIN")
    print("=" * 60)
    
    # Load dữ liệu
    daily_data = []
    with open(data_file, "r") as file:
        lines = file.read().splitlines()
        for i, line in enumerate(lines):
            if line.strip():
                numbers = [int(x.strip()) for x in line.replace(' ', ',').split(',') if x.strip()]
                daily_data.append({
                    'day': i + 1,
                    'data': numbers,
                    'count': len(numbers)
                })
    
    print(f"📅 Tổng số ngày: {len(daily_data)}")
    
    # 1. Phân tích tổng quan từng ngày
    print(f"\n📊 THỐNG KÊ TỪNG NGÀY:")
    for day_info in daily_data:
        day_data = day_info['data']
        tai_count = sum(1 for x in day_data if x > 10.5)
        xiu_count = len(day_data) - tai_count
        avg_value = sum(day_data) / len(day_data)
        
        print(f"Ngày {day_info['day']:2d}: {len(day_data):3d} kết quả | "
              f"Tài: {tai_count:2d} ({tai_count/len(day_data)*100:4.1f}%) | "
              f"Xỉu: {xiu_count:2d} | TB: {avg_value:5.2f}")
    
    # 2. Phân tích pattern theo chuỗi
    print(f"\n🎯 PHÂN TÍCH PATTERN THEO CHUỖI:")
    
    all_patterns = {}
    pattern_length = 3
    
    for day_info in daily_data:
        data = day_info['data']
        for i in range(len(data) - pattern_length):
            pattern = tuple(data[i:i+pattern_length])
            next_val = data[i+pattern_length]
            
            if pattern not in all_patterns:
                all_patterns[pattern] = []
            all_patterns[pattern].append(next_val)
    
    # Tìm pattern phổ biến và độ chính xác
    pattern_stats = []
    for pattern, next_values in all_patterns.items():
        if len(next_values) >= 2:  # Chỉ xét pattern xuất hiện ít nhất 2 lần
            avg_next = sum(next_values) / len(next_values)
            tai_count = sum(1 for x in next_values if x > 10.5)
            accuracy = max(tai_count, len(next_values) - tai_count) / len(next_values)
            
            pattern_stats.append({
                'pattern': pattern,
                'frequency': len(next_values),
                'avg_next': avg_next,
                'tai_ratio': tai_count / len(next_values),
                'accuracy': accuracy,
                'trend': 'TÀI' if avg_next > 10.5 else 'XỈU'
            })
    
    # Sắp xếp theo độ phổ biến
    pattern_stats.sort(key=lambda x: x['frequency'], reverse=True)
    
    print(f"\n🏆 TOP 10 PATTERN PHỔ BIẾN NHẤT:")
    print("Pattern       | Số lần | Tỉ lệ Tài | Độ chính xác | Xu hướng")
    print("-" * 65)
    
    for i, stat in enumerate(pattern_stats[:10]):
        pattern_str = '-'.join(map(str, stat['pattern']))
        print(f"{pattern_str:12} | {stat['frequency']:6d} | {stat['tai_ratio']:8.1%} | {stat['accuracy']:11.1%} | {stat['trend']}")
    
    # 3. Phân tích pattern đặc biệt
    print(f"\n🔍 PATTERN ĐẶC BIỆT:")
    
    # Pattern chuỗi Tài/Xỉu liên tiếp
    for day_info in daily_data:
        data = day_info['data']
        trends = ['T' if x > 10.5 else 'X' for x in data]
        
        # Tìm chuỗi dài nhất
        max_tai_streak = find_max_streak(trends, 'T')
        max_xiu_streak = find_max_streak(trends, 'X')
        
        if max_tai_streak >= 5 or max_xiu_streak >= 5:
            print(f"Ngày {day_info['day']}: Chuỗi Tài dài {max_tai_streak}, Chuỗi Xỉu dài {max_xiu_streak}")
    
    # 4. Phân tích theo khung giả định (chia ngày thành các phiên)
    print(f"\n⏰ PHÂN TÍCH THEO PHIÊN GIẢ ĐỊNH:")
    
    for day_info in daily_data[:3]:  # Phân tích 3 ngày đầu
        data = day_info['data']
        session_size = 20  # Giả định 20 kết quả/phiên
        
        for session in range(0, len(data), session_size):
            session_data = data[session:session+session_size]
            if len(session_data) >= 10:  # Chỉ xét phiên đủ dài
                tai_count = sum(1 for x in session_data if x > 10.5)
                avg_val = sum(session_data) / len(session_data)
                
                print(f"Ngày {day_info['day']} - Phiên {session//session_size + 1}: "
                      f"Tài {tai_count}/{len(session_data)} | TB: {avg_val:.2f}")
    
    # 5. Phân tích số xuất hiện nhiều nhất
    all_numbers = []
    for day_info in daily_data:
        all_numbers.extend(day_info['data'])
    
    number_counts = Counter(all_numbers)
    
    print(f"\n🔢 TOP SỐ XUẤT HIỆN NHIỀU NHẤT:")
    for number, count in number_counts.most_common(10):
        percentage = count / len(all_numbers) * 100
        trend = "TÀI" if number > 10.5 else "XỈU"
        print(f"Số {number:2d}: {count:3d} lần ({percentage:5.1f}%) - {trend}")
    
    # 6. Xuất pattern quan trọng
    export_important_patterns(pattern_stats)

def find_max_streak(trends, target):
    """Tìm chuỗi dài nhất của target trend"""
    max_streak = 0
    current_streak = 0
    
    for trend in trends:
        if trend == target:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    
    return max_streak

def export_important_patterns(pattern_stats):
    """Xuất các pattern quan trọng"""
    important_patterns = []
    
    # Pattern có độ chính xác cao
    high_accuracy = [p for p in pattern_stats if p['accuracy'] >= 0.7]
    high_frequency = [p for p in pattern_stats if p['frequency'] >= 3]
    
    print(f"\n💎 PATTERN QUAN TRỌNG:")
    print(f"• Pattern có độ chính xác ≥70%: {len(high_accuracy)}")
    print(f"• Pattern xuất hiện ≥3 lần: {len(high_frequency)}")
    
    # Xuất file pattern
    with open('sunwin_patterns.txt', 'w', encoding='utf-8') as f:
        f.write("PATTERN QUAN TRỌNG SUNWIN\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("PATTERN ĐỘ CHÍNH XÁC CAO (≥70%):\n")
        for pattern in sorted(high_accuracy, key=lambda x: x['accuracy'], reverse=True)[:10]:
            pattern_str = '-'.join(map(str, pattern['pattern']))
            f.write(f"{pattern_str} -> {pattern['trend']} (Độ chính xác: {pattern['accuracy']:.1%}, Số lần: {pattern['frequency']})\n")
        
        f.write("\nPATTERN PHỔ BIẾN (≥3 lần):\n")
        for pattern in sorted(high_frequency, key=lambda x: x['frequency'], reverse=True)[:10]:
            pattern_str = '-'.join(map(str, pattern['pattern']))
            f.write(f"{pattern_str} -> {pattern['trend']} (Số lần: {pattern['frequency']}, Độ chính xác: {pattern['accuracy']:.1%})\n")
    
    print("✅ Đã xuất pattern vào: sunwin_patterns.txt")

# Hàm chính để chạy phân tích
def run_complete_analysis():
    """Chạy phân tích hoàn chỉnh"""
    print("🔍 BẮT ĐẦU PHÂN TÍCH KẾT QUẢ THỰC TẾ...")
    df = analyze_actual_results()
    analyze_recent_performance()
    compare_with_historical()
    export_analysis_report()
    
    return df

if __name__ == "__main__":
    run_complete_analysis()