from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict()

    # 画像ディレクトリのパス
    image_dir: str = "images"
    # 出力するディレクトリのパス
    output_dir: str = "output"
    # 使用するモデル名
    model_name: str = "gpt-4o-mini"
    # モデルに渡すTemperatureの値
    temperature: float = 0.0
    # テキストの置換対象文字列
    text_replace_before: list[str] = ["...", "・・・"]
    # テキストの置換後文字列
    text_replace_after: list[str] = ["…", "…"]
    # テキストの変換対象文字
    text_trans_before: str = "·‐‑’♯　〜"
    # テキストの変換後文字
    text_trans_after: str = "・--'# ～"
    # よみがなの変換対象文字
    reading_trans_before: str = "～〜"
    # よみがなの変換後文字
    reading_trans_after: str = "ーー"
    # XのURL
    x_url: str = "https://x.com/noreal_koma/status/"
    twitter_url: str = "https://twitter.com/noreal_koma/status/"


settings = Settings()
