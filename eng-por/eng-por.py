from __future__ import division, print_function, unicode_literals

import math
import os
import random
import re
import string
import subprocess
import time
import unicodedata
import zipfile
from io import open

# import matplotlib.pyplot as plt
#plt.switch_backend('agg')  
# 日本語訳注：Google Colaboratoryでグラフが描画されなくなるため、コメントアウト
# import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists('./data'):
      print("./data/ already exists")
else:
  subprocess.run("wget https://download.pytorch.org/tutorial/data.zip",shell=True, check=True)
  with zipfile.ZipFile("./data.zip") as zipfile:
    zipfile.extractall(".")

#「Start Of Sentence（文の開始）」と「End Of Sentence（文の終了）」を表す定数を設定
SOS_token = 0
EOS_token = 1


# 言語名（name)
# 単語からインデックスへのマッピング（word2index）
# 単語の出現回数（word2count）
# インデックスから単語へのマッピング（index2word）
# 言語の単語数（n_words）
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # SOSとEOSをカウント

    # 与えられた文（sentence）を単語に分割し、各単語をLangクラスに追加
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    # 新しい単語をクラスに追加
    # 単語が既に存在しない場合は、新しいインデックスを割り当て, それに関連する情報（単語のインデックス、出現回数）が更新
    # 単語が既に存在する場合は、その単語の出現回数が増加
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
# UnicodeをASCIIに変換しています。以下のリンクを参考にしました。
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# 文字を小文字に統一し、文字以外の記号を除外しています
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # ファイルを読み込んで行に分割しています
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # 全ての行をペアに分割して正規化しています
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # ペアを逆にして、Langインスタンスを作成します
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

# できるだけ短時間でモデルを訓練させたいので比較的短くてシンプルな文だけになるよう、データセットをトリミング（切り出し）
MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


# 以下のようなシーケンスだけが取得されるようにフィルタリングしています。
# 例文の長さが10語以内（語尾の句読点を含む）
# 文の始まりは"I am"や"He is"などの形（前のステップで、アポストロフィが空白に置き換えられている短縮形のケースも含みます）
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]



def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'por', True)
print(random.choice(pairs))


# seq2seq
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        # 隠れ層のサイズをインスタンス変数として保存
        self.hidden_size = hidden_size

        # 入力データ（通常は単語のインデックス）を隠れ層のサイズの密なベクトル表現に変換するための埋め込みレイヤーを作成
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # GRU層を作成
        self.gru = nn.GRU(hidden_size, hidden_size)

    # ネットワークにデータを渡す際に使用されるメソッド
    def forward(self, input, hidden):
        # 入力データを埋め込みレイヤーでベクトル化し、適切な形状に変換
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # 入力トークンを埋め込みベクトルに変換
        self.embedding = nn.Embedding(output_size, hidden_size)
        
        # GRUレイヤーで、隠れ層の状態を更新
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
        # 出力を確率分布に変換（次に来る単語の確率を予測し、特定の文脈で最も確からしいベクトルに変換）
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # 入力トークンを埋め込みベクトルに変換し、適切な形状に変更
        output = self.embedding(input).view(1, 1, -1)
        
        # ReLU活性化関数を適用(非線形活性化関数)
        output = F.relu(output)
        # GRUレイヤーを通して、出力と新しい隠れ層の状態を計算
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        # 隠れ層のサイズ
        self.hidden_size = hidden_size
        # 出力層のサイズ（通常はターゲット言語の語彙サイズ）
        self.output_size = output_size
        # ドロップアウトの確率で、過学習を防ぐ
        self.dropout_p = dropout_p
        # 入力系列の最大長さ
        self.max_length = max_length

        # 出力語彙を埋め込むための層
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # 注意機構で使われる線形層
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        # 注意を適用した後の出力を組み合わせるための線形層
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        # アテンションレイヤーの入力や、アテンション重みを計算する際にドロップアウトを適用
        # ドロップアウトは学習時にのみ適用され、評価時には未使用（過学習の防止、過剰な依存の退避）
        # ドロップアウト層
        self.dropout = nn.Dropout(self.dropout_p)
        # GRU層
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        # 最終出力を生成するための線形層
        self.out = nn.Linear(self.hidden_size, self.output_size)

    # ネットワークにデータを通過させる際に使用
    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        
        # 重みを計算
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        
        # 重みをエンコーダー出力に適用
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        # 埋め込みベクトル, アテンションによって適用されたベクトルを結合
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        
        # 埋め込みベクトルと注意適用後のベクトルを結合
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
# training data

# 与えられた文章を単語に分割し、各単語を対応するインデックスに変換。
# このインデックスは、langオブジェクトに格納されているword2index辞書を使用して取得。
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

# indexesFromSentenceを使用して文章をインデックスのリストに変換し、さらにトークン（EOS_token、End Of Sentence）をリストに追加。
# その後、このリストをPyTorchのテンソルに変換。
def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

# 入力言語とターゲット言語の文章のペアから、それぞれの言語に対応するテンソルを生成。
# これはtensorFromSentence関数を使用する。
def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

teacher_forcing_ratio = 0.5

# 入力テンソル（input_tensor）と目標テンソル（target_tensor）を受け取り、モデルが目標テンソルにできるだけ近い出力を生成するように学習
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    # エンコーダの隠れ状態を初期化
    encoder_hidden = encoder.initHidden()

    # オプティマイザーの勾配をリセット(新しい訓練ステップを始める前)
    # エンコーダとデコーダの最適化関数（optimizer）の勾配をリセット
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # 入力と目標のテンソルの長さを取得
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # エンコーダの出力を格納するためのテンソルを初期化
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    
    # エンコーダーは入力テンソルを一つずつ処理し、それぞれのステップで隠れ状態を更新
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    # 最初のデコーダー入力はSOS（Start Of Sentence）トークン
    decoder_input = torch.tensor([[SOS_token]], device=device)
    # デコーダーの最初の隠れ状態は、エンコーダーの最終隠れ状態に設定
    decoder_hidden = encoder_hidden
    # 教師強制は、一定の確率で真（True）または偽（False）
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # 教師強制の場合：次の入力として正解データを渡す。
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
         # 教師強制を使わない：自分の予測を次の入力として使用する
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # 入力として使用するために、計算グラフから切り離す
            # 損失関数を用いてデコーダの出力と正解データとの差異を計算し、損失を蓄積
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break
    
    # 損失に基づいて勾配を計算
    # 蓄積された損失に基づいてバックプロパゲーションを行い、エンコーダとデコーダのパラメータを更新
    loss.backward()
    # エンコーダーとデコーダーのパラメータを更新
    encoder_optimizer.step()
    decoder_optimizer.step()
    # 損失の平均値
    return loss.item() / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# 与えられたエンコーダとデコーダを特定の回数（n_iters）だけ訓練
def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    # 時間計測の開始、損失のリストの初期化、および損失の合計をリセットします。
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # printする度にリセット
    plot_loss_total = 0  # plot_everyごとにリセット

    # エンコーダとデコーダのための最適化関数（SGD: 確率的勾配降下法）を設定
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # 訓練データのペアをランダムに選択して準備
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    
    # 損失関数を設定
    criterion = nn.NLLLoss()

    # 各イテレーションで、選択された訓練データペアを使用して、train 関数を呼び出します。
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            # 計算された損失を累積して、定期的に平均損失を表示
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    # 訓練が完了した後、損失の履歴をグラフとして表示
    # showPlot(plot_losses)


# def showPlot(points):
#     plt.figure()
#     fig, ax = plt.subplots()
#     # 等間隔でticを設定
#     loc = ticker.MultipleLocator(base=0.2)
#     ax.yaxis.set_major_locator(loc)
#     plt.plot(points)

# evaluate関数は、エンコーダとデコーダを使って特定の文章を評価
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        # 与えられた文章をテンソルに変換
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        # エンコーダの隠れ状態を初期化
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        # 入力テンソルをエンコーダに順番に渡し、エンコーダの出力と隠れ状態を取得
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        # 最初のデコーダの入力にはSOS_token（文章の開始を示す特別なトークン）を使用
        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        # デコーダの最初の隠れ状態は、エンコーダの最後の隠れ状態
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        # デコーダを最大長さまで実行し、各ステップで予測された単語を記録
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            # デコーダからの出力を元に次の入力を決定
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            # EOS_tokenが予測されたら、文章の終わりとみなし処理を終了
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()
            
        # デコードされた単語のリスト（翻訳された文章）と、アテンションの情報
        return decoded_words, decoder_attentions[:di + 1]

# 訓練セットからランダムに文章を取り出して評価し、
# 入力データ、正解データ、予測結果をprint文で表示
def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1, 75000, print_every=5000)

evaluateRandomly(encoder1, attn_decoder1)

output_words, attentions = evaluate(
    encoder1, attn_decoder1, "Eu sou infeliz .")
# plt.matshow(attentions.numpy())

# def showAttention(input_sentence, output_words, attentions):
#     # Set up figure with colorbar
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     cax = ax.matshow(attentions.numpy(), cmap='bone')
#     fig.colorbar(cax)

#     # Set up axes
#     ax.set_xticklabels([''] + input_sentence.split(' ') +
#                        ['<EOS>'], rotation=90)
#     ax.set_yticklabels([''] + output_words)

#     # Show label at every tick
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#     ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

#     plt.show()


# def evaluateAndShowAttention(input_sentence):
#     output_words, attentions = evaluate(
#         encoder1, attn_decoder1, input_sentence)
#     print('input =', input_sentence)
#     print('output =', ' '.join(output_words))
#     showAttention(input_sentence, output_words, attentions)


# evaluateAndShowAttention("Estou preso .")

# evaluateAndShowAttention("Eu sou infeliz .")

# evaluateAndShowAttention("Estou aqui em cima !")

# evaluateAndShowAttention("Eu estou esperando .")