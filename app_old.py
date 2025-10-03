import fasttext
import re

# Load model
model = fasttext.load_model('models/intent_model.bin')

def predict_intent(text):
    """Dự đoán intent với xử lý lỗi"""
    try:
        predictions = model.predict(text, k=1)
        intent = predictions[0][0].replace('__label__', '')
        confidence = predictions[1][0]
        return intent, confidence
    except Exception as e:
        return "UNKNOWN", 0.0

def get_action(intent):
    actions = {
        "WELCOME": """Chào bạn nha! 👋 
        Tôi có thể giúp bạn các việc sau:
        📊 **Xem lương** - Xem bảng lương cá nhân
        📅 **Xem chấm công** - Xem thông tin chấm công
        👤 **Xem thông tin cá nhân** - Xem hồ sơ cá nhân

        💡 **Ví dụ cách hỏi:**
        - "Cho tôi xem lương tháng này"
        - "Xem chấm công từ 1/10 đến 31/10"  
        - "Hiển thị thông tin cá nhân"
        - "Chấm công tháng trước"

        Hãy cho tôi biết bạn cần gì nhé!""",
        "XEM_LUONG": "Hiển thị bảng lương",
        "CHAM_CONG": "Hiển thị chấm công tháng hiện tại",
        "CHAM_CONG_THEO_NGAY": "Hiển thị chấm công theo khoảng thời gian",
        "THONG_TIN_CA_NHAN": "Hiển thị thông tin cá nhân"
    }

    return actions.get(intent, "Xin lỗi, tôi chưa hiểu yêu cầu của bạn. Hãy thử lại nhé!")

# Demo
if __name__ == "__main__":
    print("🤖 FastText AI Assistant - Nhập 'quit' để thoát\n")
    
    while True:
        user_input = input("👤 Bạn: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'thoát']:
            break
            
        if user_input:
            intent, confidence = predict_intent(user_input)
            action = get_action(intent)
            
            print(f"🤖 Bot: Intent: {intent}")
            print(f"    Độ tin cậy: {confidence:.1%}")
            print(f"    Hành động: {action}")
            print()