import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 定义情感标签映射字典
emotion_labels = {
    "平淡": 0,
    "激动": 1,
    "伤心": 2,
    "生气": 3,
    "好奇": 4,
    "厌恶": 5,
}

# 反转情感标签映射字典，用于从数字标签获取情绪名称
reverse_emotion_labels = {v: k for k, v in emotion_labels.items()}

# 加载保存的模型和分词器
save_directory = '情绪'
tokenizer = BertTokenizer.from_pretrained(save_directory)
model = BertForSequenceClassification.from_pretrained(save_directory)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# 测试函数
def test_text(text, tokenizer, model, device):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()

    return predicted_class_id

# 示例测试文本
test_texts = [
    "这不纯纯道德问题吗",
    "这不纯纯道德问题吗",
    "这不纯纯道德问题吗",
    "这不纯纯道德问题吗",
]

# 进行测试
for text in test_texts:
    result = test_text(text, tokenizer, model, device)
    emotion_name = reverse_emotion_labels[result]
    print(f"文本: {text}")
    print(f"情绪: {emotion_name}")
    print()