import numpy as np
from collections import deque
from fileInteractions import load_actual_results

class StrategyAdvisor:
    def __init__(self):
        self.bet_history = deque(maxlen=100)
        self.win_streak = 0
        self.lose_streak = 0
        
    def analyze_betting_pattern(self, actual_results):
        """Phân tích pattern đặt cược"""
        if len(actual_results) < 10:
            return "Chưa đủ dữ liệu để phân tích"
        
        results = [r['actual'] for r in actual_results]
        
        # Phân tích chu kỳ
        tai_sequence = [1 if x > 10.5 else 0 for x in results]
        
        # Tìm pattern
        patterns = self.find_patterns(tai_sequence)
        
        # Phân tích hiệu suất
        win_rate = self.calculate_win_rate(actual_results)
        
        if win_rate > 0.6:
            return "Chiến lược hiện tại HIỆU QUẢ CAO"
        elif win_rate > 0.52:
            return "Chiến lược hiện tại KHẢ QUAN"
        else:
            return "CẦN ĐIỀU CHỈNH chiến lược"
    
    def find_patterns(self, sequence):
        """Tìm các pattern trong chuỗi kết quả"""
        patterns = {}
        pattern_length = 3
        
        for i in range(len(sequence) - pattern_length):
            pattern = tuple(sequence[i:i+pattern_length])
            next_val = sequence[i+pattern_length]
            
            if pattern not in patterns:
                patterns[pattern] = []
            patterns[pattern].append(next_val)
        
        return patterns
    
    def calculate_win_rate(self, actual_results):
        """Tính tỉ lệ thắng dựa trên dự đoán"""
        if not actual_results:
            return 0.5
        
        correct_predictions = 0
        total_predictions = 0
        
        for result in actual_results:
            if 'predictions' in result and result['predictions']:
                pred_values = list(result['predictions'].values())
                if pred_values:
                    avg_pred = np.mean(pred_values)
                    pred_trend = 1 if avg_pred > 10.5 else 0
                    actual_trend = 1 if result['actual'] > 10.5 else 0
                    
                    if pred_trend == actual_trend:
                        correct_predictions += 1
                    total_predictions += 1
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.5
    
    def get_bet_suggestion(self, prediction_confidence, current_trend):
        """Đề xuất mức đặt cược"""
        if prediction_confidence > 0.7:
            return "MẠNH", 0.3  # 30% vốn
        elif prediction_confidence > 0.6:
            return "VỪA", 0.2   # 20% vốn
        elif prediction_confidence > 0.55:
            return "NHẸ", 0.1   # 10% vốn
        else:
            return "KHÔNG", 0.0  # 0% vốn
            
    def get_martingale_suggestion(self, current_streak, last_bet):
        """Đề xuất chiến lược Martingale"""
        if current_streak >= 3:
            return f"TĂNG CƯỢC: {last_bet * 2} (streak: {current_streak})"
        else:
            return f"GIỮ NGUYÊN: {last_bet}"

# Tạo instance toàn cục
strategy_advisor = StrategyAdvisor()

def calculate_win_rate_wrapper(actual_results):
    """Wrapper function để tính win rate"""
    return strategy_advisor.calculate_win_rate(actual_results)

def analyze_performance_improvement():
    """Phân tích cải thiện hiệu suất - FIXED VERSION"""
    actual_results = load_actual_results()
    
    if len(actual_results) < 10:
        return "📊 Cần thêm dữ liệu để phân tích xu hướng"
    
    # Chia thành 2 nửa để so sánh
    half = max(1, len(actual_results) // 2)
    first_half = actual_results[:half]
    second_half = actual_results[half:]
    
    # Kiểm tra xem có đủ dữ liệu dự đoán không
    first_has_predictions = any('predictions' in result and result['predictions'] for result in first_half)
    second_has_predictions = any('predictions' in result and result['predictions'] for result in second_half)
    
    if not first_has_predictions or not second_has_predictions:
        return "📈 Chưa có đủ dữ liệu dự đoán để phân tích xu hướng"
    
    # Sử dụng instance toàn cục
    first_win_rate = strategy_advisor.calculate_win_rate(first_half)
    second_win_rate = strategy_advisor.calculate_win_rate(second_half)
    
    # Tính phần trăm cải thiện
    if first_win_rate > 0:
        improvement = ((second_win_rate - first_win_rate) / first_win_rate) * 100
    else:
        improvement = second_win_rate * 100
    
    # Phân loại xu hướng
    if improvement > 15:
        return f"🚀 CẢI THIỆN MẠNH: +{improvement:.1f}%"
    elif improvement > 5:
        return f"📈 CẢI THIỆN: +{improvement:.1f}%"
    elif improvement > -5:
        return f"➡️  ỔN ĐỊNH: {improvement:+.1f}%"
    elif improvement > -15:
        return f"📉 GIẢM SÚT: {improvement:+.1f}%"
    else:
        return f"🔻 GIẢM MẠNH: {improvement:+.1f}%"

def analyze_recent_trend():
    """Phân tích xu hướng gần đây - ALTERNATIVE METHOD"""
    actual_results = load_actual_results()
    
    if len(actual_results) < 5:
        return "📊 Chưa đủ dữ liệu gần đây"
    
    # Lấy 10 kết quả gần nhất (hoặc ít hơn nếu không đủ)
    recent_results = actual_results[-min(10, len(actual_results)):]
    
    # Tính tỉ lệ thắng gần đây
    recent_win_rate = strategy_advisor.calculate_win_rate(recent_results)
    
    # So sánh với tỉ lệ thắng tổng thể
    overall_win_rate = strategy_advisor.calculate_win_rate(actual_results)
    
    if recent_win_rate > overall_win_rate + 0.1:
        return f"🎯 XU HƯỚNG TỐT: Gần đây {recent_win_rate*100:.1f}% (Tổng: {overall_win_rate*100:.1f}%)"
    elif recent_win_rate > overall_win_rate:
        return f"📈 XU HƯỚNG TÍCH CỰC: Gần đây {recent_win_rate*100:.1f}% (Tổng: {overall_win_rate*100:.1f}%)"
    elif recent_win_rate < overall_win_rate - 0.1:
        return f"⚠️  CẦN CẢI THIỆN: Gần đây {recent_win_rate*100:.1f}% (Tổng: {overall_win_rate*100:.1f}%)"
    else:
        return f"➡️  ỔN ĐỊNH: Gần đây {recent_win_rate*100:.1f}% (Tổng: {overall_win_rate*100:.1f}%)"

def get_sunwin_strategy():
    """Chiến lược đặc biệt cho dữ liệu Sunwin"""
    strategies = [
        "🎯 Chiến lược 1: Theo dõi pattern 3 số liên tiếp",
        "🎯 Chiến lược 2: Đặt ngược lại sau chuỗi 4 Tài/Xỉu liên tiếp", 
        "🎯 Chiến lược 3: Tập trung vào các số trung bình (9-12)",
        "🎯 Chiến lược 4: Chú ý các ngày có biến động mạnh",
        "🎯 Chiến lược 5: Sử dụng kết hợp pattern ngắn và dài hạn"
    ]
    
    return strategies

def calculate_expected_value(win_rate, bet_amount=1):
    """Tính giá trị kỳ vọng"""
    if win_rate >= 0.5:
        ev = (win_rate * bet_amount) - ((1 - win_rate) * bet_amount)
        return f"💰 Giá trị kỳ vọng: +{ev:.3f} trên mỗi đồng"
    else:
        ev = (win_rate * bet_amount) - ((1 - win_rate) * bet_amount)
        return f"💸 Giá trị kỳ vọng: {ev:.3f} trên mỗi đồng"

def get_performance_analysis():
    """Phân tích hiệu suất tổng hợp"""
    actual_results = load_actual_results()
    
    if not actual_results:
        return {
            'status': 'NO_DATA',
            'message': '📊 Chưa có dữ liệu để phân tích'
        }
    
    # Tính các chỉ số
    total_predictions = len(actual_results)
    win_rate = strategy_advisor.calculate_win_rate(actual_results)
    recent_trend = analyze_recent_trend()
    
    # Phân loại hiệu suất
    if win_rate > 0.6:
        performance_level = 'HIGH'
        performance_msg = '🎯 HIỆU QUẢ CAO'
    elif win_rate > 0.52:
        performance_level = 'GOOD' 
        performance_msg = '📈 KHẢ QUAN'
    elif win_rate > 0.48:
        performance_level = 'AVERAGE'
        performance_msg = '➡️  TRUNG BÌNH'
    else:
        performance_level = 'LOW'
        performance_msg = '⚠️  CẦN CẢI THIỆN'
    
    return {
        'status': 'SUCCESS',
        'total_predictions': total_predictions,
        'win_rate': win_rate,
        'win_rate_percentage': win_rate * 100,
        'performance_level': performance_level,
        'performance_message': performance_msg,
        'recent_trend': recent_trend,
        'expected_value': calculate_expected_value(win_rate)
    }

def analyze_betting_pattern_wrapper(actual_results):
    """Wrapper function cho analyze_betting_pattern"""
    return strategy_advisor.analyze_betting_pattern(actual_results)