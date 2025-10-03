import fasttext
import os

print("🚀 Training FastText với Pre-trained Vectors...")

# Kiểm tra file vectors
pretrained_path = 'models/pretrained/crawl-300d-2M.vec'
use_pretrained = os.path.exists(pretrained_path)

if use_pretrained:
    print("✅ Using pre-trained word vectors")
else:
    print("⚠️  Training from scratch (no pre-trained vectors)")

# Train model với hoặc không có pre-trained vectors
if use_pretrained:
    model = fasttext.train_supervised(
        input='data/training_data.txt',
        epoch=50,           # Có thể giảm epoch khi dùng pre-trained
        lr=0.5,
        wordNgrams=2,
        dim=300,            # Phải khớp với dimension của pre-trained vectors
        pretrainedVectors=pretrained_path,
        minCount=1,
        minn=2,
        maxn=5,
        verbose=2
    )
else:
    model = fasttext.train_supervised(
        input='data/training_data.txt',
        epoch=100,
        lr=0.5,
        wordNgrams=2,
        dim=100,            # Có thể dùng dimension nhỏ hơn
        minCount=1,
        minn=2,
        maxn=5,
        verbose=2
    )

# Lưu model
os.makedirs('models', exist_ok=True)
model.save_model('models/intent_model.bin')
print("✅ Model saved: models/intent_model.bin")

# Test khả năng hiểu ngữ nghĩa
test_cases = [
    "chào bạn",           # Test WELCOME
    "xin chào",           # Test WELCOME  
    "hello",              # Test WELCOME
    "thông tin cá nhân",
    "chấm công tháng trước",
    "chấm công tháng trước trước",
    "chấm công tháng trước trước trước",
    "xem lương tháng này",
    "chấm công hôm nay",
    "kiểm tra công tháng 8"
]

print(f"\n🧪 Testing Semantic Understanding (Pre-trained: {use_pretrained}):")
for text in test_cases:
    # Sửa lỗi predict - cách mới
    predictions = model.predict(text, k=2)
    
    # FastText trả về tuple (labels, probabilities)
    labels = predictions[0]
    probabilities = predictions[1]
    
    print(f"📝 '{text}'")
    for i in range(len(labels)):
        intent = labels[i].replace('__label__', '')
        prob = probabilities[i]
        print(f"   → {intent}: {prob:.1%}")
    print()