import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 定义情感标签映射字典
emotion_labels = {
    0: "0",
    1: "1",

}

# 加载保存的模型和分词器
save_directory = '记忆'
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

    return emotion_labels[predicted_class_id]

# 示例测试文本
test_texts = [
    "我是邓紫棋",
    "你买了的嗯。",
    "这是什么",
    "瞧这是啥",
    "邓紫棋就是我",
    "你看这是什么",
]

# 进行测试
for text in test_texts:
    result = test_text(text, tokenizer, model, device)
    print(f"文本: {text}")
    print(f"{result}")
    print()
    