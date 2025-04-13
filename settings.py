from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # OpenAI APIキー
    openai_api_key: str
    # 画像ディレクトリのパス
    image_dir: str = "images"
    # 出力するディレクトリのパス
    output_dir: str = "output"
    # 使用するモデル名
    model_name: str = "gpt-4o-mini"
    # モデルに渡すTemperatureの値
    temperature: float = 0.0
    # テキストの変換対象文字
    text_trans_before: str = "·‐‑’♯　〜"
    # テキストの変換後文字
    text_trans_after: str = "・--'# ～"
    # よみがなの変換対象文字
    reading_trans_before: str = "～〜"
    # よみがなの変換後文字
    reading_trans_after: str = "ーー"


settings = Settings()
