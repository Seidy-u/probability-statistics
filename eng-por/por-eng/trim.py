# ファイルを開いて読み込む
with open('por.txt', 'r') as file:
    lines = file.readlines()

# 各行を処理する
processed_lines = []
for line in lines:
    # 2つのタブで分割し、最初の2つの部分を取得
    parts = line.split('\t', 2)
    processed_line = parts[0] + '\t' + parts[1] if len(parts) > 1 else parts[0]
    processed_lines.append(processed_line)

# 結果を新しいファイルに書き出す
with open('por-eng.txt', 'w') as file:
    for line in processed_lines:
        file.write(line + '\n')
