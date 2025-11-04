import numpy as np
import random
from collections import Counter
from fileInteractions import save_data, exportConfig, load_data_file
import os

class DataEnhancer:
    def __init__(self):
        self.config = exportConfig()
        
    def enhance_with_patterns(self, auto_merge=True):
        """Tăng cường dữ liệu dựa trên pattern phân tích được"""
        print("🚀 Bắt đầu tăng cường dữ liệu với pattern...")
        
        # Đảm bảo file dữ liệu tồn tại
        data_file = load_data_file()
        print(f"📁 File dữ liệu: {data_file}")
        
        # Đầu tiên, tạo dữ liệu cơ bản từ file gốc
        base_data = self.load_base_data()
        if base_data:
            print(f"📥 Đã tải {len(base_data)} chuỗi từ dữ liệu gốc")
        
        # Pattern từ phân tích
        high_accuracy_patterns = [
            # Pattern có độ chính xác cao
            ([15, 9, 10], 8),   # -> Xỉu
            ([9, 10, 12], 7),   # -> Xỉu  
            ([9, 14, 10], 15),  # -> Tài
            ([11, 8, 9], 14),   # -> Tài
            ([8, 12, 7], 11),   # -> Tài
            ([10, 13, 9], 8),   # -> Xỉu
            ([13, 9, 10], 12),  # -> Tài
        ]
        
        # Số xuất hiện nhiều nhất từ phân tích
        common_numbers = [11, 9, 10, 12, 8, 7, 13, 14, 6, 15]
        
        enhanced_data = []
        
        # 1. Thêm dữ liệu gốc
        enhanced_data.extend(base_data)
        
        # 2. Tạo dữ liệu từ pattern chính xác
        print("📊 Tạo dữ liệu từ pattern chính xác...")
        for pattern, next_val in high_accuracy_patterns:
            # Tạo biến thể từ pattern gốc
            for _ in range(10):  # Tăng số biến thể
                variant = self.create_pattern_variant(pattern, common_numbers)
                enhanced_data.append(variant + [next_val])
        
        # 3. Tạo dữ liệu từ chuỗi Tài/Xỉu dài
        print("📈 Tạo dữ liệu từ chuỗi xu hướng...")
        enhanced_data.extend(self.generate_trend_sequences())
        
        # 4. Tạo dữ liệu từ phân phối số thực tế
        print("🎲 Tạo dữ liệu từ phân phối số...")
        enhanced_data.extend(self.generate_distribution_sequences(common_numbers))
        
        # 5. Tạo dữ liệu từ phân tích phiên
        print("⏰ Tạo dữ liệu từ phân tích phiên...")
        enhanced_data.extend(self.generate_session_based_sequences())
        
        # Loại bỏ trùng lặp
        unique_data = []
        seen = set()
        for seq in enhanced_data:
            seq_tuple = tuple(seq)
            if seq_tuple not in seen:
                seen.add(seq_tuple)
                unique_data.append(seq)
        
        enhanced_data = unique_data
        
        # Lưu dữ liệu tăng cường
        if enhanced_data:
            # Đảm bảo thư mục tồn tại
            enhanced_file = "enhanced_data.txt"
            os.makedirs(os.path.dirname(enhanced_file) if os.path.dirname(enhanced_file) else ".", exist_ok=True)
            
            # Xóa file cũ nếu tồn tại
            if os.path.exists(enhanced_file):
                os.remove(enhanced_file)
                
            for sequence in enhanced_data:
                save_data(sequence, enhanced_file)
            
            print(f"✅ Đã tạo {len(enhanced_data)} chuỗi dữ liệu tăng cường")
            print(f"📁 Lưu vào: {enhanced_file}")
            
            # Thống kê dữ liệu tăng cường
            self.analyze_enhanced_data(enhanced_data)
            
            # Tự động merge nếu được yêu cầu
            if auto_merge:
                self.merge_enhanced_data()
                print("🔄 Đã tự động merge vào dữ liệu chính")
        else:
            print("❌ Không tạo được dữ liệu tăng cường")
        
        return enhanced_data
    
    def load_base_data(self):
        """Tải dữ liệu cơ bản từ file gốc"""
        base_data = []
        config = exportConfig()
        data_file = config["data_file"]
        
        # Đảm bảo file tồn tại
        if not os.path.exists(data_file):
            print(f"⚠️  File {data_file} không tồn tại, tạo file mới...")
            os.makedirs(os.path.dirname(data_file) if os.path.dirname(data_file) else ".", exist_ok=True)
            with open(data_file, 'w') as f:
                # Thêm dữ liệu mẫu cơ bản
                sample_data = [
                    [11, 16, 5, 7, 11, 10, 9, 12],
                    [10, 10, 13, 11, 14, 9, 15, 6],
                    [12, 16, 10, 10, 7, 8, 9, 12],
                    [12, 7, 8, 12, 7, 9, 9, 14]
                ]
                for seq in sample_data:
                    f.write(",".join(map(str, seq)) + "\n")
            print(f"✅ Đã tạo file {data_file} với dữ liệu mẫu")
        
        try:
            with open(data_file, "r", encoding='utf-8') as file:
                lines = file.read().splitlines()
                for line in lines:
                    if line.strip():
                        # Xử lý nhiều định dạng
                        line_clean = line.replace(' ', ',').replace('\t', ',')
                        numbers = [int(x.strip()) for x in line_clean.split(',') if x.strip()]
                        if len(numbers) >= 5:  # Giảm yêu cầu độ dài
                            base_data.append(numbers)
        except Exception as e:
            print(f"❌ Lỗi khi đọc file {data_file}: {e}")
        
        return base_data
    
    def create_pattern_variant(self, base_pattern, common_numbers):
        """Tạo biến thể từ pattern gốc"""
        variant = base_pattern.copy()
        
        # Thay đổi ngẫu nhiên 1-2 số trong pattern
        num_changes = random.randint(0, 2)  # Có thể không thay đổi
        for _ in range(num_changes):
            change_pos = random.randint(0, len(variant) - 1)
            # Thay bằng số phổ biến khác
            available_nums = [n for n in common_numbers if n != variant[change_pos]]
            if available_nums:
                new_num = random.choice(available_nums)
                variant[change_pos] = new_num
        
        return variant
    
    def generate_trend_sequences(self):
        """Tạo chuỗi dựa trên xu hướng Tài/Xỉu"""
        sequences = []
        
        # Tạo nhiều chuỗi hơn
        for _ in range(20):  # Giảm số lượng để tránh quá nhiều
            seq_length = random.randint(6, 10)  # Giảm độ dài
            seq = []
            
            # Chọn xu hướng ban đầu
            current_trend = random.choice(['tai', 'xiu'])
            
            for i in range(seq_length):
                # Có 20% khả năng đổi trend
                if random.random() < 0.2:
                    current_trend = 'xiu' if current_trend == 'tai' else 'tai'
                
                if current_trend == 'tai':
                    seq.append(random.randint(11, 16))
                else:
                    seq.append(random.randint(5, 10))
            sequences.append(seq)
        
        return sequences
    
    def generate_distribution_sequences(self, common_numbers):
        """Tạo chuỗi dựa trên phân phối số thực tế"""
        sequences = []
        
        # Tần suất số từ phân tích
        number_weights = {
            11: 150, 9: 150, 10: 140, 12: 140, 8: 120,
            7: 120, 13: 100, 14: 90, 6: 70, 15: 60,
            5: 40, 16: 35, 17: 25, 18: 20, 4: 15, 3: 10
        }
        
        numbers = list(number_weights.keys())
        weights = list(number_weights.values())
        
        for _ in range(30):  # Giảm số lượng
            seq_length = random.randint(6, 9)
            seq = []
            for i in range(seq_length):
                num = random.choices(numbers, weights=weights)[0]
                seq.append(num)
            sequences.append(seq)
        
        return sequences
    
    def generate_session_based_sequences(self):
        """Tạo chuỗi dựa trên phân tích phiên"""
        sequences = []
        
        # Tạo nhiều loại phiên khác nhau
        session_types = [
            ('tai_strong', 0.7, 10),   # Phiên Tài mạnh
            ('xiu_strong', 0.3, 10),   # Phiên Xỉu mạnh  
            ('balanced', 0.5, 15),     # Phiên cân bằng
            ('volatile', 0.45, 10)     # Phiên biến động
        ]
        
        for session_type, tai_prob, count in session_types:
            for _ in range(count):
                seq_length = random.randint(6, 9)
                seq = []
                for i in range(seq_length):
                    if random.random() < tai_prob:
                        seq.append(random.randint(11, 16))
                    else:
                        seq.append(random.randint(5, 10))
                sequences.append(seq)
        
        return sequences
    
    def analyze_enhanced_data(self, enhanced_data):
        """Phân tích dữ liệu tăng cường"""
        print(f"\n📈 PHÂN TÍCH DỮ LIỆU TĂNG CƯỜNG")
        print("=" * 40)
        
        if not enhanced_data:
            print("❌ Không có dữ liệu để phân tích")
            return
            
        all_numbers = []
        for seq in enhanced_data:
            all_numbers.extend(seq)
        
        # Thống kê cơ bản
        total_numbers = len(all_numbers)
        tai_count = sum(1 for x in all_numbers if x > 10.5)
        xiu_count = total_numbers - tai_count
        
        print(f"Tổng số: {total_numbers}")
        print(f"Tài: {tai_count} ({tai_count/total_numbers*100:.1f}%)")
        print(f"Xỉu: {xiu_count} ({xiu_count/total_numbers*100:.1f}%)")
        print(f"Trung bình: {np.mean(all_numbers):.2f}")
        
        # Top số xuất hiện
        number_counts = Counter(all_numbers)
        print(f"\n🔢 TOP SỐ XUẤT HIỆN:")
        for num, count in number_counts.most_common(8):
            percentage = count / total_numbers * 100
            trend = "TÀI" if num > 10.5 else "XỈU"
            print(f"Số {num:2d}: {count:3d} lần ({percentage:5.1f}%) - {trend}")
    
    def merge_enhanced_data(self):
        """Merge dữ liệu tăng cường vào file chính"""
        try:
            enhanced_file = "enhanced_data.txt"
            if not os.path.exists(enhanced_file):
                print(f"❌ Không tìm thấy file {enhanced_file}")
                return
                
            with open(enhanced_file, "r", encoding='utf-8') as f:
                enhanced_lines = f.read().splitlines()
            
            # Đọc dữ liệu hiện có để tránh trùng lặp
            existing_data = set()
            config = exportConfig()
            data_file = config["data_file"]
            
            # Đảm bảo file tồn tại
            if not os.path.exists(data_file):
                print(f"⚠️  File {data_file} không tồn tại, tạo mới...")
                os.makedirs(os.path.dirname(data_file) if os.path.dirname(data_file) else ".", exist_ok=True)
                with open(data_file, 'w', encoding='utf-8') as f:
                    pass
            
            if os.path.exists(data_file):
                with open(data_file, "r", encoding='utf-8') as f:
                    for line in f:
                        existing_data.add(line.strip())
            
            # Thêm dữ liệu mới
            added_count = 0
            with open(data_file, "a", encoding='utf-8') as f:
                for line in enhanced_lines:
                    if line.strip() and line not in existing_data:
                        if added_count == 0 and len(existing_data) > 0:
                            f.write("\n" + line)
                        else:
                            f.write(line + "\n")
                        added_count += 1
            
            print(f"✅ Đã merge {added_count} chuỗi mới vào dữ liệu chính")
            
        except Exception as e:
            print(f"❌ Lỗi khi merge dữ liệu: {e}")

# Hàm tiện ích
def enhance_training_data():
    """Chạy tăng cường dữ liệu"""
    try:
        enhancer = DataEnhancer()
        
        # Hỏi người dùng có muốn auto merge không
        auto_merge_input = input("🤔 Có tự động merge vào dữ liệu chính không? (y/n) [mặc định: y]: ").strip()
        auto_merge = auto_merge_input.lower() != 'n' if auto_merge_input else True
        
        enhanced_data = enhancer.enhance_with_patterns(auto_merge=auto_merge)
        
        if enhanced_data and auto_merge:
            print("🔁 Chạy 'enhanced_train' để huấn luyện với dữ liệu mới")
            
        return True
        
    except Exception as e:
        print(f"❌ Lỗi trong quá trình tăng cường dữ liệu: {e}")
        return False

# Tạo dữ liệu mẫu nhanh
def create_quick_sample_data():
    """Tạo dữ liệu mẫu nhanh để test"""
    sample_data = [
        [11, 16, 5, 7, 11, 10, 9, 12],
        [10, 10, 13, 11, 14, 9, 15, 6],
        [12, 16, 10, 10, 7, 8, 9, 12],
        [12, 7, 8, 12, 7, 9, 9, 14],
        [7, 12, 13, 11, 12, 9, 11, 8],
        [3, 7, 9, 8, 12, 8, 10, 13],
        [9, 10, 17, 8, 6, 5, 9, 9],
        [11, 8, 8, 7, 13, 8, 8, 10]
    ]
    
    config = exportConfig()
    data_file = config["data_file"]
    
    # Đảm bảo thư mục tồn tại
    os.makedirs(os.path.dirname(data_file) if os.path.dirname(data_file) else ".", exist_ok=True)
    
    with open(data_file, 'w', encoding='utf-8') as f:
        for seq in sample_data:
            f.write(",".join(map(str, seq)) + "\n")
    
    print(f"✅ Đã tạo file {data_file} với {len(sample_data)} chuỗi mẫu")
    return sample_data

if __name__ == "__main__":
    # Kiểm tra xem có cần tạo dữ liệu mẫu không
    config = exportConfig()
    data_file = config["data_file"]
    
    if not os.path.exists(data_file):
        print("📝 Tạo dữ liệu mẫu...")
        create_quick_sample_data()
    
    enhance_training_data()