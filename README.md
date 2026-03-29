# noreal-koma-ocr

OpenAI または Ollama の Vision モデルを用いて
存在しない漫画の1コマbot ([@noreal_koma](https://x.com/noreal_koma))
の漫画からテキストデータを抽出する

## 実行に必要なパッケージ

- jaconv
- langchain
- langchain-openai
- pydantic-settings

## 実行方法

### ローカル実行

#### OpenAI を使用する場合

```bash
python main.py
```

#### Ollama を使用する場合

```bash
python main.py --ollama
```

### Docker Compose での実行

実行前に UID を指定してビルドする

```bash
docker compose build --build-arg UID=$(id -u)
```

#### OpenAI を使用する場合

```bash
docker compose up
```

#### Ollama を使用する場合 (ホストの Ollama に接続)

```bash
docker compose run -e OLLAMA_BASE_URL="http://host.docker.internal:11434/v1" app python main.py --ollama
```

## コマンドライン引数

- `--ollama`: ローカルのOllama APIを使用する
- `--model`: 使用するモデル名を指定する (例: `--model qwen3.5:9b`)

## 実行前に設定が必要な環境変数

OpenAI を使用する場合は API キーの設定が必要

```env
# OpenAI APIキー
OPENAI_API_KEY=OpenAI APIキー
```
