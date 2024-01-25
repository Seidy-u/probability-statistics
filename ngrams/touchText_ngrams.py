import torch
import torchtext
from torchtext.datasets import text_classification
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from torch.utils.data.dataset import random_split
import re
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer


# データセット：
# 1 : 国際関連
# 2 : スポーツ 
# 3 : ビジネス 
# 4 : 科学技術

NGRAMS = 2
if not os.path.isdir('./.data'):
	os.mkdir('./.data')
# PyTorchが提供するテキスト分類のための標準データセットである「AG_NEWS」を読み込み
# ngrams=NGRAMSで使用するn-gramのサイズを設定。
# vocab=Noneは、既存の語彙を使用せずにデータセットから語彙。
train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
    root='./.data', ngrams=NGRAMS, vocab=None)
# 一度にネットワークに渡されるデータサンプルの数
BATCH_SIZE = 16
# CUDAが利用可能な場合はGPUを使用　そうでない場合はCPUを使用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        # vocab_size: 語彙のサイズ（使用される単語の総数）
        # embed_dim: 埋め込みベクトルの次元数
        # num_class: 分類するクラスの数
        super().__init__()
        # 埋め込みと集約（bagging）テキストの単語IDを入力として受け取り、それらの単語の埋め込みベクトルの平均または合計を出力
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        # 埋め込みベクトルをクラスの予測スコアに変換
        self.fc = nn.Linear(embed_dim, num_class)
        # ネットワークの重みを初期化
        self.init_weights()

    # 埋め込みレイヤーと線形レイヤーの両方の重みをランダムな値で初期化し、線形レイヤーのバイアスはゼロで初期化
    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    # text: 入力テキストの単語IDのテンソル。
    # offsets: 各テキストの開始位置を示すオフセットのテンソル
    # EmbeddingBagにはテキストのバッチを処理する機能があるため、どこで各テキストの埋め込みを開始および終了するかをモデルが知るために必要
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)
    
# トレーニングデータセットの語彙のサイズ（単語の総数）を取得
VOCAB_SIZE = len(train_dataset.get_vocab())
# 各単語の埋め込みベクトルの次元数を32に設定
EMBED_DIM = 32
# トレーニングデータセットに含まれるラベル（クラス）の数を取得
# モデルの出力層のサイズを決定するのに使用され、分類するクラスの数に対応
NUN_CLASS = len(train_dataset.get_labels())
# TextSentimentモデルを初期化
# .to(device)は、モデルを指定されたデバイス（CPUまたはGPU）に移動
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)


# EmbeddingBagレイヤー
# batch: データセットから取得されたバッチ。
# batchはgenerate_batch関数の引数として宣言
# この関数がDataLoaderによって呼び出されるとき、DataLoaderはデータセットから取得した一連のデータ（バッチ）を自動的にbatch引数としてgenerate_batch関数に渡す
def generate_batch(batch):
    # バッチ内の各エントリから抽出し、テンソルに変換
    label = torch.tensor([entry[0] for entry in batch])
    # バッチ内の各エントリからテキストデータを抽出
    text = [entry[1] for entry in batch]
    # 各テキストの長さに基づいてオフセット（各テキストの開始インデックス）を計算
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsumはdimで指定されたの要素の累積和を返します。
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    # オフセットのリストを累積和に変換してテンソルに変換
    # これにより、各テキストの開始位置を取得
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    # すべてのテキストデータを一つの長いテンソルに結合
    text = torch.cat(text)
    return text, offsets, label

def train_func(sub_train_):
    
    # モデル訓練
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=generate_batch)
    
    # テキスト、オフセット、クラスラベルをデバイスに移動。
    # モデルで予測を行い、損失を計算。
    # 正解率を計算。
    for i, (text, offsets, cls) in enumerate(data):
        # 最適化されるパラメーターの勾配をリセット
        # 勾配は、パラメーターを更新する際に使用されるため、各トレーニングステップの開始時にリセット
        optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        # モデルにテキストデータとオフセットを入力として与え、出力（予測）を取得
        output = model(text, offsets)
        loss = criterion(output, cls)
        # 現在のバッチにおける損失を累積損失に加算
        train_loss += loss.item()
        # 計算された損失に基づいて、モデルのパラメーターに対する勾配を計算
        loss.backward()
        # 最適化関数を使用して、モデルのパラメーターを更新
        optimizer.step()
        # モデルの出力のうち、最も確率の高いクラス（output.argmax(1)）が実際のラベル（cls）と一致しているかをチェックし、正解数を累積
        train_acc += (output.argmax(1) == cls).sum().item()

    # 学習率の更新
    scheduler.step()

    # 平均損失と平均正解率
    return train_loss / len(sub_train_), train_acc / len(sub_train_)


def test(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(data_), acc / len(data_)

# 総エポック数を定義
N_EPOCHS = 5
min_valid_loss = float('inf')

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

train_len = int(len(train_dataset) * 0.95)
sub_train_, sub_valid_ = \
    random_split(train_dataset, [train_len, len(train_dataset) - train_len])

for epoch in range(N_EPOCHS):

    start_time = time.time()
    train_loss, train_acc = train_func(sub_train_)
    valid_loss, valid_acc = test(sub_valid_)

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')
    
print('Checking the results of test dataset...')
test_loss, test_acc = test(test_dataset)
print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')

ag_news_label = {1 : "World",
                 2 : "Sports",
                 3 : "Business",
                 4 : "Sci/Tec"}

# 与えられたテキストを受け取り、モデルを使用してカテゴリーを予測
def predict(text, model, vocab, ngrams):
    # tokenizer でテキストをトークンに分割し、ngrams_iterator でN-gramを生成
    tokenizer = get_tokenizer("basic_english")
    with torch.no_grad():
        text = torch.tensor([vocab[token]
                            for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1


ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    enduring the season’s worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he’d never played the \
    front nine at TPC Southwind."


vocab = train_dataset.get_vocab()
model = model.to("cpu")

print("This is a %s news" %ag_news_label[predict(ex_text_str, model, vocab, 2)])

# 日本語訳注：上記はゴルフ関連の記事文章なので、Sportsに分類されて欲しいです。