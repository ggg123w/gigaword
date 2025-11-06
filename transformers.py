import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
import os
import json
from tqdm import tqdm
import warnings
import pickle
import time
import random

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 忽略字体警告
warnings.filterwarnings("ignore", category=UserWarning)

# 设置matplotlib不使用中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 创建数据目录
os.makedirs("data", exist_ok=True)

# 加载数据集到指定目录 - 使用更多数据
print("加载数据集...")
dataset = load_dataset("Gabriel/gigaword_swe", cache_dir="data")

# 使用更多数据
sample_size = 20000  # 增加数据量
train_dataset = dataset['train'].select(range(min(sample_size, len(dataset['train']))))
val_dataset = dataset['validation'].select(range(min(2000, len(dataset['validation']))))
test_dataset = dataset['test'].select(range(min(2000, len(dataset['test']))))

print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 查看数据样例
print("\n数据样例:")
print("文档:", train_dataset[0]['document'][:100] + "...")
print("摘要:", train_dataset[0]['summary'][:100] + "...")

# 构建词汇表
class Vocabulary:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.vocab_size = 4
        
    def build_vocab(self, texts, min_freq=3):  # 进一步提高频率阈值
        word_freq = {}
        for text in texts:
            words = text.split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        for word, freq in word_freq.items():
            if freq >= min_freq:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
    
    def encode(self, text, max_length=128):
        words = text.split()[:max_length]
        indices = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        return [self.word2idx['<SOS>']] + indices + [self.word2idx['<EOS>']]
    
    def decode(self, indices):
        words = []
        for idx in indices:
            if idx == self.word2idx['<EOS>']:
                break
            if idx not in [self.word2idx['<PAD>'], self.word2idx['<SOS>']]:
                words.append(self.idx2word.get(idx, '<UNK>'))
        return ' '.join(words)
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f)
    
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.__dict__.update(pickle.load(f))

# 检查是否已有保存的词汇表
vocab_path = 'data/vocab.pkl'
if os.path.exists(vocab_path):
    print("加载已保存的词汇表...")
    vocab = Vocabulary()
    vocab.load(vocab_path)
    print(f"词汇表大小: {vocab.vocab_size}")
else:
    print("构建词汇表...")
    vocab = Vocabulary()
    
    # 使用抽样数据构建词汇表
    all_documents = [example['document'] for example in train_dataset]
    all_summaries = [example['summary'] for example in train_dataset]
    all_texts = all_documents + all_summaries
    
    vocab.build_vocab(all_texts, min_freq=3)  # 提高频率阈值
    vocab.save(vocab_path)
    print(f"词汇表大小: {vocab.vocab_size} (已保存)")

# 数据集类
class TextDataset(Dataset):
    def __init__(self, dataset, vocab, max_length=128):
        self.dataset = dataset
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        document = self.dataset[idx]['document']
        summary = self.dataset[idx]['summary']
        
        # 编码文本
        doc_encoded = self.vocab.encode(document, self.max_length)
        sum_encoded = self.vocab.encode(summary, self.max_length)
        
        # 填充序列
        doc_padded = self.pad_sequence(doc_encoded, self.max_length)
        sum_padded = self.pad_sequence(sum_encoded, self.max_length)
        
        return {
            'document': torch.tensor(doc_padded, dtype=torch.long),
            'summary': torch.tensor(sum_padded, dtype=torch.long),
            'doc_length': min(len(doc_encoded), self.max_length),
            'sum_length': min(len(sum_encoded), self.max_length)
        }
    
    def pad_sequence(self, sequence, max_length):
        if len(sequence) < max_length:
            return sequence + [self.vocab.word2idx['<PAD>']] * (max_length - len(sequence))
        else:
            return sequence[:max_length]

# 创建数据加载器
batch_size = 32
max_length = 64

print("创建数据加载器...")
train_loader = DataLoader(
    TextDataset(train_dataset, vocab, max_length),
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    TextDataset(val_dataset, vocab, max_length),
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    TextDataset(test_dataset, vocab, max_length),
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# 多头注意力 - 添加更多正则化
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.2):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(dropout)  # 注意力dropout
        
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            mask = mask.to(attn_scores.dtype)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)  # 应用注意力dropout
        
        output = torch.matmul(attn_weights, v)
        return output, attn_weights
    
    def forward(self, q, k, v, mask=None):
        batch_size, seq_len = q.size(0), q.size(1)
        
        # 线性变换并分头
        q = self.w_q(q).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, k.size(1), self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, v.size(1), self.num_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        attn_output, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 输出线性变换
        output = self.w_o(attn_output)
        return output, attn_weights

# 位置前馈网络 - 添加更多正则化
class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.2):
        super(PositionWiseFFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout1 = nn.Dropout(dropout)  # 第一个dropout
        self.dropout2 = nn.Dropout(dropout)  # 第二个dropout
        self.activation = nn.GELU()
    
    def forward(self, x):
        return self.linear2(self.dropout2(self.activation(self.dropout1(self.linear1(x)))))

# 编码器层 - 增加dropout
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.2):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFFN(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力 + 残差 + LayerNorm
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差 + LayerNorm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

# 解码器层 - 增加dropout
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.2):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFFN(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 自注意力 + 残差 + LayerNorm
        self_attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # 交叉注意力 + 残差 + LayerNorm
        cross_attn_output, _ = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 前馈网络 + 残差 + LayerNorm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

# 编码器
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_length, dropout=0.2):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_length)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # 通过所有编码器层
        for layer in self.layers:
            x = layer(x, mask)
        
        return x

# 解码器
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_length, dropout=0.2):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_length)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # 通过所有解码器层
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        
        return x

# 完整的Transformer模型 - 增加更多正则化
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, num_layers=3, 
                 num_heads=8, d_ff=256, max_length=128, dropout=0.3):  # 增加模型容量和dropout
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, max_length, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_length, dropout)
        self.final_linear = nn.Linear(d_model, tgt_vocab_size)
        
        # 添加额外的dropout
        self.output_dropout = nn.Dropout(dropout)
        
        # 参数初始化
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_mask(self, src, tgt):
        # 源序列掩码 (用于编码器和解码器的交叉注意力)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        # 目标序列掩码 (用于解码器的自注意力)
        tgt_len = tgt.size(1)
        tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        nopeak_mask = (1 - torch.triu(torch.ones(1, tgt_len, tgt_len), diagonal=1)).bool().to(device)
        tgt_mask = tgt_padding_mask & nopeak_mask
        
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        # 确保tgt有足够的长度
        if tgt.size(1) <= 1:
            batch_size = src.size(0)
            return torch.zeros(batch_size, 1, self.final_linear.out_features).to(device)
        
        src_mask, tgt_mask = self.create_mask(src, tgt[:, :-1])
        
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt[:, :-1], enc_output, src_mask, tgt_mask)
        
        # 应用输出dropout
        dec_output = self.output_dropout(dec_output)
        
        output = self.final_linear(dec_output)
        return output
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# 创建模型
model = Transformer(
    src_vocab_size=vocab.vocab_size,
    tgt_vocab_size=vocab.vocab_size,
    d_model=128,  # 增加模型容量
    num_layers=3,  # 增加层数
    num_heads=8,   # 增加头数
    d_ff=256,      # 增加前馈网络维度
    max_length=max_length,
    dropout=0.3    # 增加dropout
).to(device)

print(f"模型参数量: {model.count_parameters():,}")

# 标签平滑损失函数 - 另一种正则化技术
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1, ignore_index=0):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            mask = (target == self.ignore_index).unsqueeze(1).expand_as(true_dist)
            true_dist.masked_fill_(mask, 0)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# 训练配置
num_epochs = 20  # 增加训练轮数
learning_rate = 1e-4  # 降低学习率

# 优化器和损失函数 - 使用标签平滑
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)  # 增加权重衰减
criterion = LabelSmoothingLoss(classes=vocab.vocab_size, smoothing=0.1, ignore_index=vocab.word2idx['<PAD>'])

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# 训练和验证函数
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="训练")
    
    for i, batch in enumerate(progress_bar):
        src = batch['document'].to(device, non_blocking=True)
        tgt = batch['summary'].to(device, non_blocking=True)
        
        # 前向传播
        output = model(src, tgt)
        loss = criterion(output.contiguous().view(-1, output.size(-1)), 
                        tgt[:, 1:].contiguous().view(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="验证"):
            src = batch['document'].to(device, non_blocking=True)
            tgt = batch['summary'].to(device, non_blocking=True)
            
            output = model(src, tgt)
            loss = criterion(output.contiguous().view(-1, output.size(-1)), 
                            tgt[:, 1:].contiguous().view(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

# 训练循环 - 只进行训练和验证，不进行测试
print("开始训练...")
train_losses = []
val_losses = []

best_val_loss = float('inf')
patience = 5
counter = 0

start_time = time.time()

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 50)
    
    # 训练
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    train_losses.append(train_loss)
    
    # 验证
    val_loss = evaluate(model, val_loader, criterion)
    val_losses.append(val_loss)
    
    # 学习率调度
    scheduler.step(val_loss)
    
    print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
    print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
    
    # 早停检查
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'vocab': vocab
        }, 'data/best_model.pth')
        print("保存最佳模型!")
    else:
        counter += 1
        if counter >= patience:
            print(f"早停: 验证损失在 {patience} 个epoch内没有改善")
            break

end_time = time.time()
print(f"总训练时间: {(end_time - start_time)/60:.2f} 分钟")

# 保存训练历史
training_history = {
    'train_losses': train_losses,
    'val_losses': val_losses
}

with open('data/training_history.json', 'w') as f:
    json.dump(training_history, f)

# 绘制训练曲线
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(val_losses, label='Val Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('data/training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n训练完成!")
print(f"最终训练损失: {train_losses[-1]:.4f}")
print(f"最终验证损失: {val_losses[-1]:.4f}")

def translate(model, text, vocab, max_length=64):
    model.eval()
    with torch.no_grad():
        # 编码输入文本
        encoded = vocab.encode(text, max_length)
        src = torch.tensor([encoded], dtype=torch.long).to(device)
        
        # 创建源序列掩码
        src_mask = (src != vocab.word2idx['<PAD>']).unsqueeze(1).unsqueeze(2)
        
        # 编码器前向传播
        enc_output = model.encoder(src, src_mask)
        
        # 开始符号
        tgt_input = torch.tensor([[vocab.word2idx['<SOS>']]], dtype=torch.long).to(device)
        
        # 自回归生成
        for i in range(max_length):
            # 创建目标序列掩码
            tgt_len = tgt_input.size(1)
            tgt_padding_mask = (tgt_input != vocab.word2idx['<PAD>']).unsqueeze(1).unsqueeze(3)
            nopeak_mask = (1 - torch.triu(torch.ones(1, tgt_len, tgt_len), diagonal=1)).bool().to(device)
            tgt_mask = tgt_padding_mask & nopeak_mask
            
            # 解码器前向传播
            dec_output = model.decoder(tgt_input, enc_output, src_mask, tgt_mask)
            output = model.final_linear(dec_output)
            
            # 获取最后一个token的预测
            next_token_logits = output[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            # 添加到目标序列
            tgt_input = torch.cat([tgt_input, next_token.unsqueeze(0)], dim=1)
            
            # 检查是否生成了EOS
            if next_token.item() == vocab.word2idx['<EOS>']:
                break
        
        # 解码输出 - 去掉<SOS>
        generated = tgt_input[0, 1:].cpu().tolist()
        return vocab.decode(generated)

# 简化的推理函数（备用方案）
def simple_translate(model, text, vocab, max_length=64):
    model.eval()
    with torch.no_grad():
        # 编码输入文本
        encoded = vocab.encode(text, max_length)
        src = torch.tensor([encoded], dtype=torch.long).to(device)
        
        # 开始符号
        tgt_ids = [vocab.word2idx['<SOS>']]
        
        for i in range(max_length):
            tgt = torch.tensor([tgt_ids], dtype=torch.long).to(device)
            
            # 使用模型的前向传播
            output = model(src, tgt)
            next_token_logits = output[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            
            tgt_ids.append(next_token_id)
            
            if next_token_id == vocab.word2idx['<EOS>']:
                break
        
        # 解码输出 - 去掉<SOS>
        return vocab.decode(tgt_ids[1:])

# 加载最佳模型进行测试
print("\n加载最佳模型进行测试...")
checkpoint = torch.load('data/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 在测试集上评估最佳模型
print("在测试集上评估最佳模型...")
test_loss = evaluate(model, test_loader, criterion)
print(f"最佳模型测试损失: {test_loss:.4f}")

# 测试几个例子
print("\n测试几个例子:")
success_count = 0
for i in range(5):
    sample = test_dataset[i]
    original_text = sample['document']
    reference_summary = sample['summary']
    
    try:
        generated_summary = translate(model, original_text, vocab)
        
        print(f"\n示例 {i+1}:")
        print(f"原文: {original_text[:100]}...")
        print(f"参考摘要: {reference_summary}")
        print(f"生成摘要: {generated_summary}")
        print("-" * 80)
        success_count += 1
        
    except Exception as e:
        print(f"示例 {i+1} 推理失败: {e}")
        continue

    
print("\n生成的文件包括:")
print("- data/best_model.pth (最佳模型)")
print("- data/training_history.json (训练历史)")
print("- data/training_curves.png (训练曲线图)")
print("- data/vocab.pkl (词汇表)")