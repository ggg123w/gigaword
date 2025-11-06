import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
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
print(f"Using device: {device}")

# 创建数据目录
os.makedirs("data", exist_ok=True)

# 加载数据集
print("Loading dataset...")
dataset = load_dataset("Gabriel/gigaword_swe", cache_dir="data")

# 使用完整数据集
print(f"Full dataset size - Train: {len(dataset['train'])}, Validation: {len(dataset['validation'])}")

# 创建训练集的子集
train_indices = list(range(len(dataset['train'])))
selected_indices = random.sample(train_indices, len(train_indices) // 4)

train_dataset = Subset(dataset['train'], selected_indices)
val_dataset = dataset['validation']
test_dataset = dataset['test']


# 查看数据样例
print("\nData sample:")
print("Document:", dataset['train'][0]['document'][:100] + "...")
print("Summary:", dataset['train'][0]['summary'][:100] + "...")

# 构建词汇表
class Vocabulary:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.vocab_size = 4
        
    def build_vocab(self, texts, min_freq=3):
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
    print("Loading saved vocabulary...")
    vocab = Vocabulary()
    vocab.load(vocab_path)
    print(f"Vocabulary size: {vocab.vocab_size}")
else:
    print("Building vocabulary...")
    vocab = Vocabulary()
    
    # 使用完整训练数据构建词汇表
    sample_texts = [example['document'] for example in dataset['train'].select(range(100000))]
    sample_texts += [example['summary'] for example in dataset['train'].select(range(100000))]
    
    vocab.build_vocab(sample_texts, min_freq=3)
    vocab.save(vocab_path)
    print(f"Vocabulary size: {vocab.vocab_size} (saved)")

# 数据集类
class TextDataset(Dataset):
    def __init__(self, dataset, vocab, max_length=128):
        self.dataset = dataset
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # 处理Subset和原始dataset的区别
        if hasattr(self.dataset, 'dataset'):
            # 如果是Subset
            data_item = self.dataset.dataset[self.dataset.indices[idx]]
        else:
            # 如果是原始dataset
            data_item = self.dataset[idx]
            
        document = data_item['document']
        summary = data_item['summary']
        
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
batch_size = 128
max_length = 128

print("Creating data loaders...")
train_loader = DataLoader(
    TextDataset(train_dataset, vocab, max_length),
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

val_loader = DataLoader(
    TextDataset(val_dataset, vocab, max_length),
    batch_size=batch_size,
    shuffle=False,
    num_workers=8,
    pin_memory=True
)

test_loader = DataLoader(
    TextDataset(test_dataset, vocab, max_length),
    batch_size=batch_size,
    shuffle=False,
    num_workers=8,
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

# 空位置编码（用于消融实验）
class NoPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(NoPositionalEncoding, self).__init__()
    
    def forward(self, x):
        return x

# 多头注意力
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
        self.attention_dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            mask = mask.to(attn_scores.dtype)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        
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

# 位置前馈网络
class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.2):
        super(PositionWiseFFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        return self.linear2(self.dropout2(self.activation(self.dropout1(self.linear1(x)))))

# 编码器层
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

# 解码器层
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

# 编码器 - 修改以支持无位置编码
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_length, dropout=0.2, use_positional_encoding=True):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.use_positional_encoding = use_positional_encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 根据参数选择是否使用位置编码
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(d_model, max_length)
        else:
            self.pos_encoding = NoPositionalEncoding(d_model, max_length)
            
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 嵌入
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # 位置编码
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # 通过所有编码器层
        for layer in self.layers:
            x = layer(x, mask)
        
        return x

# 解码器 - 修改以支持无位置编码
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_length, dropout=0.2, use_positional_encoding=True):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.use_positional_encoding = use_positional_encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 根据参数选择是否使用位置编码
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(d_model, max_length)
        else:
            self.pos_encoding = NoPositionalEncoding(d_model, max_length)
            
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 嵌入
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # 位置编码
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # 通过所有解码器层
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        
        return x

# 完整的Transformer模型 - 修改以支持无位置编码
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, num_layers=3, 
                 num_heads=8, d_ff=256, max_length=128, dropout=0.3, use_positional_encoding=True):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, max_length, dropout, use_positional_encoding)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_length, dropout, use_positional_encoding)
        self.final_linear = nn.Linear(d_model, tgt_vocab_size)
        
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
    d_model=128,
    num_layers=3,
    num_heads=8,
    d_ff=256,
    max_length=max_length,
    dropout=0.3,
    use_positional_encoding=True
).to(device)

print(f"Model parameters: {model.count_parameters():,}")

# 标签平滑损失函数
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
num_epochs = 20
learning_rate = 3e-4

# 优化器和损失函数
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
criterion = LabelSmoothingLoss(classes=vocab.vocab_size, smoothing=0.1, ignore_index=vocab.word2idx['<PAD>'])

# 学习率调度器
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# 训练和验证函数
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")
    
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
        for batch in tqdm(dataloader, desc="Validation"):
            src = batch['document'].to(device, non_blocking=True)
            tgt = batch['summary'].to(device, non_blocking=True)
            
            output = model(src, tgt)
            loss = criterion(output.contiguous().view(-1, output.size(-1)), 
                            tgt[:, 1:].contiguous().view(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

# 训练循环
print("Starting training...")
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
    scheduler.step()
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")
    print(f"Learning rate: {current_lr:.6f}")
    
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
        print("Saved best model!")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping: Validation loss did not improve for {patience} epochs")
            break

end_time = time.time()
print(f"Total training time: {(end_time - start_time)/60:.2f} minutes")

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

print("\nTraining completed!")
print(f"Final training loss: {train_losses[-1]:.4f}")
print(f"Final validation loss: {val_losses[-1]:.4f}")

# 消融实验函数
def run_ablation_study():
    """Run ablation study to compare different model configurations"""
    print("\n" + "="*60)
    print("Starting Ablation Study")
    print("="*60)
    
    # 使用0.1%的数据进行消融实验
    ablation_sample_size = int(len(dataset['train']) * 0.001)
    print(f"Using {ablation_sample_size} samples for ablation study")
    
    # 创建抽样数据集
    ablation_train_indices = random.sample(range(len(dataset['train'])), ablation_sample_size)
    ablation_train_dataset = Subset(dataset['train'], ablation_train_indices)
    ablation_val_dataset = dataset['validation'].select(range(min(1000, len(dataset['validation']))))
    
    # 创建数据加载器
    ablation_train_loader = DataLoader(
        TextDataset(ablation_train_dataset, vocab, max_length),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    
    ablation_val_loader = DataLoader(
        TextDataset(ablation_val_dataset, vocab, max_length),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    # 消融实验配置
    ablation_configs = {
        'baseline': {
            'd_model': 128,
            'num_layers': 3,
            'num_heads': 8,
            'd_ff': 256,
            'dropout': 0.3,
            'label_smoothing': 0.1,
            'use_positional_encoding': True
        },
        'no_positional_encoding': {
            'd_model': 128,
            'num_layers': 3,
            'num_heads': 8,
            'd_ff': 256,
            'dropout': 0.3,
            'label_smoothing': 0.1,
            'use_positional_encoding': False
        },
        'reduced_heads_4': {
            'd_model': 128,
            'num_layers': 3,
            'num_heads': 4,
            'd_ff': 256,
            'dropout': 0.3,
            'label_smoothing': 0.1,
            'use_positional_encoding': True
        },
        'increased_heads_16': {
            'd_model': 128,
            'num_layers': 3,
            'num_heads': 16,
            'd_ff': 256,
            'dropout': 0.3,
            'label_smoothing': 0.1,
            'use_positional_encoding': True
        }
    }
    
    ablation_results = {}
    
    for config_name, config in ablation_configs.items():
        print(f"\nTraining configuration: {config_name}")
        print(f"Parameters: {config}")
        
        # 创建模型
        model = Transformer(
            src_vocab_size=vocab.vocab_size,
            tgt_vocab_size=vocab.vocab_size,
            d_model=config['d_model'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            d_ff=config['d_ff'],
            max_length=max_length,
            dropout=config['dropout'],
            use_positional_encoding=config['use_positional_encoding']
        ).to(device)
        
        print(f"Model parameters: {model.count_parameters():,}")
        
        # 优化器和损失函数
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        criterion = LabelSmoothingLoss(
            classes=vocab.vocab_size, 
            smoothing=config['label_smoothing'], 
            ignore_index=vocab.word2idx['<PAD>']
        )
        
        # 简化的训练循环
        num_ablation_epochs = 5
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(num_ablation_epochs):
            train_loss = train_epoch(model, ablation_train_loader, optimizer, criterion)
            val_loss = evaluate(model, ablation_val_loader, criterion)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            print(f"Epoch {epoch+1}/{num_ablation_epochs}: Train loss={train_loss:.4f}, Val loss={val_loss:.4f}")
        
        # 在测试集上评估
        test_loss = evaluate(model, test_loader, criterion)
        
        ablation_results[config_name] = {
            'config': config,
            'parameters': model.count_parameters(),
            'best_val_loss': best_val_loss,
            'test_loss': test_loss,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        print(f"{config_name} - Best validation loss: {best_val_loss:.4f}, Test loss: {test_loss:.4f}")
    
    # 保存消融实验结果
    with open('data/ablation_results.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)
    
    # 绘制消融实验结果
    plot_ablation_results(ablation_results)
    
    return ablation_results

def plot_ablation_results(results):
    """Plot ablation study results"""
    config_names = list(results.keys())
    val_losses = [results[name]['best_val_loss'] for name in config_names]
    test_losses = [results[name]['test_loss'] for name in config_names]
    parameters = [results[name]['parameters'] for name in config_names]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Validation loss comparison
    bars1 = ax1.bar(config_names, val_losses, color='skyblue', alpha=0.7)
    ax1.set_title('Ablation Study - Validation Loss Comparison')
    ax1.set_ylabel('Validation Loss')
    ax1.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars1, val_losses):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Test loss comparison
    bars2 = ax2.bar(config_names, test_losses, color='lightcoral', alpha=0.7)
    ax2.set_title('Ablation Study - Test Loss Comparison')
    ax2.set_ylabel('Test Loss')
    ax2.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars2, test_losses):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Parameter count comparison
    bars3 = ax3.bar(config_names, parameters, color='lightgreen', alpha=0.7)
    ax3.set_title('Ablation Study - Model Parameter Count Comparison')
    ax3.set_ylabel('Parameter Count')
    ax3.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars3, parameters):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val*0.01, 
                f'{val:,}', ha='center', va='bottom', fontsize=9)
    
    # Loss comparison scatter plot
    ax4.scatter(val_losses, test_losses, s=100, alpha=0.6)
    for i, name in enumerate(config_names):
        ax4.annotate(name, (val_losses[i], test_losses[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax4.set_xlabel('Validation Loss')
    ax4.set_ylabel('Test Loss')
    ax4.set_title('Validation Loss vs Test Loss')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/ablation_study_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot attention heads comparison
    plot_attention_heads_comparison(results)

def plot_attention_heads_comparison(results):
    """Plot comparison of different attention head counts"""
    head_configs = {
        'reduced_heads_4': 4,
        'baseline': 8,
        'increased_heads_16': 16
    }
    
    # Filter out attention head related configurations
    head_results = {k: v for k, v in results.items() if k in head_configs}
    
    if not head_results:
        return
    
    head_nums = [head_configs[name] for name in head_results.keys()]
    val_losses = [head_results[name]['best_val_loss'] for name in head_results.keys()]
    test_losses = [head_results[name]['test_loss'] for name in head_results.keys()]
    parameters = [head_results[name]['parameters'] for name in head_results.keys()]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Validation loss vs attention heads
    ax1.plot(head_nums, val_losses, 'o-', linewidth=2, markersize=8, label='Validation Loss')
    ax1.set_xlabel('Number of Attention Heads')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Effect of Attention Heads on Validation Loss')
    ax1.grid(True, alpha=0.3)
    
    # Test loss vs attention heads
    ax2.plot(head_nums, test_losses, 'o-', linewidth=2, markersize=8, color='orange', label='Test Loss')
    ax2.set_xlabel('Number of Attention Heads')
    ax2.set_ylabel('Test Loss')
    ax2.set_title('Effect of Attention Heads on Test Loss')
    ax2.grid(True, alpha=0.3)
    
    # Parameter count vs attention heads
    ax3.plot(head_nums, parameters, 'o-', linewidth=2, markersize=8, color='green', label='Parameter Count')
    ax3.set_xlabel('Number of Attention Heads')
    ax3.set_ylabel('Parameter Count')
    ax3.set_title('Effect of Attention Heads on Parameter Count')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/attention_heads_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# 运行消融实验
ablation_results = run_ablation_study()

print("\n" + "="*60)
print("All experiments completed!")
print("="*60)

# 输出总结
print("\nGenerated files:")
print("- data/best_model.pth (Best model)")
print("- data/training_history.json (Training history)")
print("- data/training_curves.png (Training curves)")
print("- data/vocab.pkl (Vocabulary)")
print("- data/ablation_results.json (Ablation study results)")
print("- data/ablation_study_results.png (Ablation study charts)")
print("- data/attention_heads_comparison.png (Attention heads comparison)")

# 显示消融实验总结
print("\nAblation Study Summary:")
for config_name, result in ablation_results.items():
    print(f"{config_name}: Validation loss={result['best_val_loss']:.4f}, Test loss={result['test_loss']:.4f}, Parameters={result['parameters']:,}")

# 分析位置编码的影响
if 'baseline' in ablation_results and 'no_positional_encoding' in ablation_results:
    baseline_loss = ablation_results['baseline']['test_loss']
    no_pos_loss = ablation_results['no_positional_encoding']['test_loss']
    pos_encoding_impact = (no_pos_loss - baseline_loss) / baseline_loss * 100
    
    print(f"\nPositional Encoding Impact Analysis:")
    print(f"With positional encoding test loss: {baseline_loss:.4f}")
    print(f"Without positional encoding test loss: {no_pos_loss:.4f}")
    print(f"Performance change: {pos_encoding_impact:.2f}%")
    
    if pos_encoding_impact > 0:
        print("Conclusion: Positional encoding has a significant positive impact on model performance")
    else:
        print("Conclusion: Positional encoding has limited impact on model performance")

# 分析注意力头数的影响
if 'reduced_heads_4' in ablation_results and 'baseline' in ablation_results and 'increased_heads_16' in ablation_results:
    heads_4_loss = ablation_results['reduced_heads_4']['test_loss']
    heads_8_loss = ablation_results['baseline']['test_loss']
    heads_16_loss = ablation_results['increased_heads_16']['test_loss']
    
    print(f"\nAttention Heads Impact Analysis:")
    print(f"4 attention heads test loss: {heads_4_loss:.4f}")
    print(f"8 attention heads test loss: {heads_8_loss:.4f}")
    print(f"16 attention heads test loss: {heads_16_loss:.4f}")
    
    # 找出最佳配置
    best_heads = min([(heads_4_loss, 4), (heads_8_loss, 8), (heads_16_loss, 16)], key=lambda x: x[0])
    print(f"Optimal number of attention heads: {best_heads[1]} (Test loss: {best_heads[0]:.4f})")
