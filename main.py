from mlModels import train, test, predict, ktest
from enhanced_mlModels import train_enhanced_models, predict_enhanced
from analysis import run_complete_analysis, analyze_patterns
from fileInteractions import exportConfig, updateConfig, save_data, save_actual_result, load_actual_results
from data_enhancer import enhance_training_data
from strategy_advisor import get_performance_analysis, get_sunwin_strategy
import sys

def show_help():
    """Hiển thị hướng dẫn sử dụng"""
    print("""
🎲 HỆ THỐNG DỰ ĐOÁN TÀI XỈU SUNWIN - HƯỚNG DẪN SỬ DỤNG 🎲

CÁC LỆNH:
---------
help           - Hiển thị hướng dẫn này
train          - Huấn luyện models cơ bản
enhanced_train - Huấn luyện models nâng cao
test           - Kiểm tra độ chính xác
predict        - Dự đoán số tiếp theo
enhanced_predict - Dự đoán nâng cao
config         - Xem/cập nhật cấu hình
analysis       - Phân tích kết quả thực tế
pattern        - Phân tích pattern dữ liệu
stats          - Xem thống kê nhanh
export         - Xuất báo cáo phân tích
enhance        - Tăng cường dữ liệu training
strategy       - Xem phân tích chiến lược
add <số1,số2>  - Thêm dữ liệu mới
ktest          - Kiểm tra kernel SVR
exit           - Thoát chương trình

CÁCH SỬ DỤNG:
-------------
>> predict all 8 9 9 11 9 9 8 10
→ Dự đoán với 8 số [8,9,9,11,9,9,8,10] dùng tất cả models

>> enhanced_predict 8 9 9 11 9 9 8 10
→ Dự đoán nâng cao với features phong phú

>> pattern
→ Phân tích pattern thường gặp trong dữ liệu

SAU KHI DỰ ĐOÁN:
---------------
Hệ thống sẽ hiển thị "kết quả đúng: " 
→ Nhập số thực tế để lưu và cải thiện model
""")

def show_config():
    """Hiển thị cấu hình hiện tại"""
    config = exportConfig()
    print("\n⚙️ CẤU HÌNH HIỆN TẠI:")
    print(f"📁 File dữ liệu    : {config['data_file']}")
    print(f"🔢 Độ dài đầu vào  : {config['input_length']}")
    print(f"🌲 Số cây RF       : {config['trees_in_the_forest']}")
    print(f"🔍 Kernel SVR      : {config['kernel']}")

def update_config_interactive():
    """Cập nhật cấu hình tương tác"""
    config = exportConfig()
    print("\n✏️ CẬP NHẬT CẤU HÌNH:")
    
    try:
        new_length = int(input(f"Độ dài đầu vào [{config['input_length']}]: ") or config['input_length'])
        new_trees = int(input(f"Số cây Random Forest [{config['trees_in_the_forest']}]: ") or config['trees_in_the_forest'])
        new_kernel = input(f"Kernel SVR [{config['kernel']}]: ") or config['kernel']
        new_file = input(f"File dữ liệu [{config['data_file']}]: ") or config['data_file']
        
        config.update({
            'input_length': new_length,
            'trees_in_the_forest': new_trees,
            'kernel': new_kernel,
            'data_file': new_file
        })
        updateConfig(config)
        print("✅ Đã cập nhật cấu hình!")
    except ValueError:
        print("❌ Giá trị không hợp lệ!")

def main():
    print("🎲 CHÀO MỪNG ĐẾN HỆ THỐNG DỰ ĐOÁN TÀI XỈU SUNWIN! 🎲")
    print("📍 Gõ 'help' để xem hướng dẫn sử dụng")
    
    while True:
        try:
            command = input("\n>> ").strip()
            
            if command == 'exit':
                print("👋 Tạm biệt!")
                break
                
            elif command == 'help':
                show_help()
                
            elif command == 'train':
                print("🔄 Bắt đầu huấn luyện models cơ bản...")
                train()
                
            elif command == 'enhanced_train':
                print("🚀 Bắt đầu huấn luyện models nâng cao...")
                try:
                    success = train_enhanced_models()
                    if success:
                        print("✅ Huấn luyện models nâng cao thành công!")
                    else:
                        print("❌ Huấn luyện thất bại! Hãy chạy 'enhance' trước")
                except ImportError as e:
                    print(f"❌ Lỗi: {e}")
                    print("💡 Hãy chạy: pip install lightgbm")
                
            elif command == 'test':
                print("🧪 Nhập dữ liệu test (ví dụ: 11,12,9,10,13,8,11,10):")
                test_input = input("Dữ liệu test: ").strip()
                try:
                    test_data = [int(x) for x in test_input.split(",")]
                    test(test_data)
                except ValueError:
                    print("❌ Dữ liệu không hợp lệ!")
                    
            elif command.startswith('predict'):
                parts = command.split()
                if len(parts) < 3:
                    print("❌ Thiếu dữ liệu! Ví dụ: predict all 8 9 9 11 9 9 8 10")
                    continue
                    
                algorithm = parts[1]
                data_input = parts[2:]
                
                try:
                    input_data = [int(x) for x in data_input]
                    
                    # Gọi hàm predict và nhận kết quả
                    result = predict(input_data, algorithm)
                    
                    # Nếu predict thành công và trả về dict (có chứa input data)
                    if result and isinstance(result, dict):
                        # Đợi người dùng nhập kết quả thực tế
                        actual_input = input()
                        if actual_input.strip():
                            try:
                                actual_result = int(actual_input.strip())
                                # Lưu kết quả thực tế
                                save_actual_result(input_data, actual_result)
                                print(f"✅ Đã lưu kết quả thực tế: {actual_result}")
                                print("🔁 Chạy 'train' để cập nhật model với dữ liệu mới")
                            except ValueError:
                                print("❌ Kết quả không hợp lệ, bỏ qua")
                        
                except ValueError:
                    print("❌ Dữ liệu không hợp lệ!")
                    
            elif command.startswith('enhanced_predict'):
                parts = command.split()
                if len(parts) < 2:
                    print("❌ Thiếu dữ liệu! Ví dụ: enhanced_predict 8 9 9 11 9 9 8 10")
                    continue
                    
                data_input = parts[1:]
                try:
                    input_data = [int(x) for x in data_input]
                    
                    # Gọi hàm predict nâng cao
                    result = predict_enhanced(input_data, "enhanced")
                    
                    if result and isinstance(result, dict):
                        actual_input = input()
                        if actual_input.strip():
                            try:
                                actual_result = int(actual_input.strip())
                                save_actual_result(input_data, actual_result)
                                print(f"✅ Đã lưu kết quả thực tế: {actual_result}")
                                
                                # Phân tích chiến lược
                                try:
                                    analysis = get_performance_analysis()
                                    if analysis['status'] == 'SUCCESS':
                                        print(f"🎯 PHÂN TÍCH CHIẾN LƯỢC: {analysis['performance_message']}")
                                        print(f"📊 XU HƯỚNG: {analysis['recent_trend']}")
                                except ImportError:
                                    print("⚠️  Không thể load phân tích chiến lược")
                                
                            except ValueError:
                                print("❌ Kết quả không hợp lệ")
                                
                except ValueError:
                    print("❌ Dữ liệu không hợp lệ!")
                    
            elif command == 'config':
                show_config()
                change = input("\nMuốn thay đổi cấu hình? (y/n): ").lower()
                if change == 'y':
                    update_config_interactive()
                    
            elif command == 'analysis':
                print("🔍 Bắt đầu phân tích kết quả thực tế...")
                try:
                    run_complete_analysis()
                except ImportError:
                    print("❌ Không thể import module analysis")
                    
            elif command == 'pattern':
                print("🎯 Phân tích pattern dữ liệu Sunwin...")
                try:
                    analyze_patterns()
                except ImportError:
                    print("❌ Không thể import module analysis")
                    
            elif command == 'stats':
                try:
                    from fileInteractions import get_analysis_stats
                    stats = get_analysis_stats()
                    print(f"\n📊 THỐNG KÊ NHANH:")
                    print(f"• Tổng kết quả: {stats['total']}")
                    print(f"• Tài: {stats['tai_count']} ({stats['tai_percentage']:.1f}%)")
                    print(f"• Xỉu: {stats['xiu_count']} ({stats['xiu_percentage']:.1f}%)")
                except ImportError:
                    print("❌ Không thể import module analysis")

            elif command == 'export':
                try:
                    from analysis import export_analysis_report
                    export_analysis_report()
                except ImportError:
                    print("❌ Không thể import module analysis")
                    
            elif command == 'enhance':
                print("🚀 Tăng cường dữ liệu training...")
                try:
                    success = enhance_training_data()
                    if success:
                        print("✅ Tăng cường dữ liệu thành công!")
                    else:
                        print("❌ Tăng cường dữ liệu thất bại!")
                except Exception as e:
                    print(f"❌ Lỗi: {e}")
                    
            elif command == 'strategy':
                try:
                    analysis = get_performance_analysis()
                    
                    if analysis['status'] == 'NO_DATA':
                        print(analysis['message'])
                    else:
                        print(f"\n🎯 PHÂN TÍCH CHIẾN LƯỢC TOÀN DIỆN")
                        print("=" * 50)
                        print(f"• Tổng số dự đoán: {analysis['total_predictions']}")
                        print(f"• Tỉ lệ thắng: {analysis['win_rate_percentage']:.1f}%")
                        print(f"• Đánh giá: {analysis['performance_message']}")
                        print(f"• {analysis['expected_value']}")
                        print(f"• Xu hướng: {analysis['recent_trend']}")
                        
                        # Hiển thị chiến lược đề xuất
                        print(f"\n💡 CHIẾN LƯỢC ĐỀ XUẤT:")
                        strategies = get_sunwin_strategy()
                        for i, strategy in enumerate(strategies[:3], 1):
                            print(f"  {strategy}")
                            
                except ImportError as e:
                    print(f"❌ Không thể load module phân tích: {e}")
                    
            elif command.startswith('add'):
                parts = command.split()
                if len(parts) < 2:
                    print("❌ Thiếu dữ liệu! Ví dụ: add 11,12,9,10,13")
                    continue
                    
                data_input = parts[1]
                try:
                    new_data = [int(x) for x in data_input.split(",")]
                    save_data(new_data)
                    print("✅ Đã thêm dữ liệu thành công!")
                except ValueError:
                    print("❌ Dữ liệu không hợp lệ!")
                    
            elif command == 'ktest':
                print("🔍 Kiểm tra kernel SVR:")
                print("Nhập dữ liệu train file (mặc định: assets/sunwin.txt):")
                train_file = input("File train: ").strip() or "assets/sunwin.txt"
                print("Nhập dữ liệu test (ví dụ: 11,12,9,10,13,8):")
                test_input = input("Dữ liệu test: ").strip()
                try:
                    test_data = [int(x) for x in test_input.split(",")]
                    ktest(train_file, test_data)
                except ValueError:
                    print("❌ Dữ liệu không hợp lệ!")
                    
            else:
                print("❌ Lệnh không hợp lệ! Gõ 'help' để xem hướng dẫn")
                
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    main()