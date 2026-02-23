def inference(image_name):
    model.eval()
    img_path = os.path.join(IMG_DIR, image_name)
    if not os.path.exists(img_path): return "File not found"
    
    img = Image.open(img_path).convert('RGB')
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_t = test_transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        probs = torch.sigmoid(model(img_t)).cpu().numpy()[0]
    
    attr_names = ["Attr1", "Attr2", "Attr3", "Attr4"]
    present = [attr_names[i] for i, p in enumerate(probs) if p > 0.5]
    print(f"Result for {image_name}: {present} | Scores: {probs.round(3)}")

# Verification
inference('Path_to_specific_image')
