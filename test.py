try:
    import fasttext
    print("✅ FastText installed successfully!")
    # print(f"Version: {fasttext.__version__}")
    
    # Test cơ bản
    model = fasttext.train_supervised(input='', epoch=1)
    print("✅ FastText is working!")
    
except Exception as e:
    print(f"❌ Error: {e}")