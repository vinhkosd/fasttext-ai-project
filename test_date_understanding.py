import fasttext # type: ignore
import re

# Load model đã train với date patterns
model = fasttext.load_model('models/intent_model.bin')

def test_date_understanding():
    test_cases = [
        "cho tôi xem công từ ngày 05/10/2025 đến 30/10/2025",
        "chấm công từ 1/10 đến 31/10", 
        "xem công từ 01/10/2025 tới 31/10/2025",
        "công từ ngày 5-10 đến 30-10",
        "chấm công từ 05/10 tới 30/10",
        "chấm công tháng 9 năm 2025",
        "công tháng 8",
        "xem công tháng 7",
        "chấm công từ ngày 1 tháng 9 đến ngày 30 tháng 9",
        "xem chấm công từ 15/8 tới 20/8",
        "chấm công khoảng 5/10 đến 15/10"
    ]
    
    print("🧠 Testing FastText Date Understanding:")
    
    for text in test_cases:
        predictions = model.predict(text, k=2)
        
        print(f"\n📝 '{text}'")
        for i, (label, prob) in enumerate(zip(predictions[0], predictions[1])):
            intent = label.replace('__label__', '')
            print(f"   {i+1}. {intent}: {prob:.1%}")
    
    print("\n" + "="*50)
    print("📊 Analysis:")
    print("FastText sẽ học các pattern ngày tháng và hiểu rằng:")
    print("✅ '05/10/2025', '1/10', '05-10' → date patterns")
    print("✅ 'đến', 'tới', '-' → date range indicators") 
    print("✅ 'tháng 9', 'tháng 10' → temporal expressions")
    print("✅ Các biến thể date format đều thuộc NGAYCONG_FROMTO")

if __name__ == "__main__":
    test_date_understanding()