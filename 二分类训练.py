import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

loda_data="记忆.csv" # 替换为你的数据文件路径
save_path="记忆"  # 替换为你想保存模型的路径
# 自定义数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# 加载数据
# 加载数据
# 加载数据
def load_data(file_path):
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        next(file)  # 跳过标题行
        for line in file:
            line = line.strip()
            if line:
                parts = line.split(',', 1)  # 只分割一次
                if len(parts) == 2:
                    text, label = parts
                    texts.append(text)
                    labels.append(int(label))
                else:
                    print(f"跳过格式错误的行: {line}")
    return texts, labels


# 训练函数
def train(model, train_dataloader, optimizer, device, epochs):
    model.train()
    try:
        for epoch in range(epochs):
            total_loss = 0
            # 使用 tqdm 为训练批次添加进度条
            progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch')
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

                # 更新进度条显示的平均损失
                avg_loss = total_loss / (progress_bar.n + 1)
                progress_bar.set_postfix({'Average Loss': avg_loss})

            avg_loss = total_loss / len(train_dataloader)
            print(f'Epoch {epoch + 1}/{epochs}, Final Average Loss: {avg_loss}')
    except KeyboardInterrupt:
        print("训练被手动终止，正在保存当前模型...")


# 主函数
def main():
    # 配置参数
    file_path = loda_data # 替换为你的数据文件路径
    model_name = 'chinese-pert-large'
    max_length = 128
    batch_size = 16
    epochs = 3
    learning_rate = 1e-4

    # 加载数据
    texts, labels = load_data(file_path)

    # 划分训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # 加载分词器和模型
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(set(labels)))

    # 创建数据集和数据加载器
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 设备配置
    device = torch.device('cuda')
    model.to(device)

    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # 训练模型
    train(model, train_dataloader, optimizer, device, epochs)

    # 保存模型和分词器
    save_directory = save_path
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"模型已保存至 {save_directory}")


if __name__ == "__main__":
    main()
    