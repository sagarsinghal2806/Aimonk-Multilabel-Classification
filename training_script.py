device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = AimonkDataset(LABEL_PATH, IMG_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
p_weights = get_pos_weights(LABEL_PATH).to(device)

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 4)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
history = {'iteration': [], 'loss': []}

for epoch in range(12):
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = masked_weighted_loss(model(images), labels, p_weights)
        loss.backward(); optimizer.step()
        history['iteration'].append(len(history['iteration'])); history['loss'].append(loss.item())
    print(f"Epoch {epoch+1}/12 | Loss: {loss.item():.4f}")

torch.save(model.state_dict(), 'deep-model.pth')

plt.figure(figsize=(8, 4))
plt.plot(history['iteration'], history['loss'])
plt.ylabel('training_loss'); plt.xlabel('iteration_number'); plt.title('Aimonk_multilabel_problem')
plt.savefig('loss_curve.png'); plt.show()
