import argparse
import base64
import json
import logging
import mimetypes
import os
import re
import textwrap
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

from jaconv import kata2hira
from langchain.chains import TransformChain
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from settings import settings


class ImageInformation(BaseModel):
    """取得する画像情報のモデル"""

    text: list[str] = Field(
        description=textwrap.dedent(
            """
                文字列の抽出
                - 範囲: 吹き出し内のセリフ、および欄外の注意書き・煽り文を含める
                - 除外:
                  - 擬声語・擬音語(オノマトペ)は抽出対象外とする
                  - ルビ(ふりがな)は無視し、親文字のみを抽出する
                - 順序: 日本の漫画の一般的な読み順(右上 → 右下 → 左上 → 左下)に従って配列に格納する
                - 粒度: 1つの吹き出し(またはテキストブロック)ごとに1つの文字列とみなし、各ブロックをリストの1要素とする
                - 書式: 吹き出し(またはテキストブロック)内部の改行は削除する
                - 誤字・脱字：修正せず、画像にあるがままを出力する (意図的な表現である可能性を考慮)
                - 記号:
                  - 句読点(、。)：全角として出力
                  - 感嘆符・疑問符(！？)：
                    - 連続する場合は半角(!!や!?など)として出力
                    - 単体であれば全角(！、？)として出力
            """
        ).strip()
    )

    reading: list[str] = Field(
        description=textwrap.dedent(
            """
                よみがなの付与
                - textの各要素に対応する読みをひらがなで提供する
                - textの要素1つにつき、1つの読みが対応するリストとする
                - 読みが分からない場合は空の文字列をリストの1要素とする
            """
        ).strip()
    )

    tag: list[str] = Field(
        description=textwrap.dedent(
            """
                視覚要素タグ
                - textに含まれない視覚的な情報を抽出する
                - 基準: pixivのタグとして一般的に存在し、検索に活用される用語を選択する
                - 対象: 服装、髪型、小道具、背景の特徴など
                - 禁止事項: 「キャラクター」や「吹き出し」、「モノクロ」など、漫画自体の概念そのものをタグとして含めないこと
                - タグとして出力できるものがない場合は空のリストを返す
            """
        ).strip()
    )


def clean_text(text) -> str:
    """
    以下の処理を行う
    - 指定された文字列を置換
    - 指定された文字を変換
    - 結合文字等を正規化
    """
    for before, after in zip(settings.text_replace_before, settings.text_replace_after):
        text = text.replace(before, after)

    return unicodedata.normalize(
        "NFC",
        text.translate(
            str.maketrans(
                settings.text_trans_before,
                settings.text_trans_after,
                "",
            )
        ),
    )


def clean_reading(text) -> str:
    """
    以下の処理を行う
    - 指定された文字を変換
    - カタカナをひらがなに変換
    - 結合文字等を正規化
    - ひらがな以外を削除
    """
    return re.sub(
        r"[^\u3040-\u309Fー]",
        "",
        unicodedata.normalize(
            "NFC",
            kata2hira(
                text.translate(
                    str.maketrans(
                        settings.reading_trans_before,
                        settings.reading_trans_after,
                        "",
                    )
                )
            ),
        ),
    )


def encode_image(image_path: str) -> str:
    """画像をBase64エンコードし、MIMEタイプを付与して返す"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError("Unsupported file type")

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded_image}"


def load_image(inputs: dict) -> dict:
    """画像をBase64エンコードし、URLとして返す"""
    image_path = inputs.get("image_path")
    if not image_path:
        raise ValueError("image_path is required")

    return {"image_url": encode_image(image_path)}


@chain
def image_model(inputs: dict) -> str | list[str | dict]:
    """画像とプロンプトを用いてモデルを呼び出す"""
    if settings.use_ollama:
        model = ChatOpenAI(
            model=settings.model_name,
            temperature=settings.temperature,
            base_url=settings.ollama_base_url,
            api_key="ollama",
        )
    else:
        model = ChatOpenAI(
            model=settings.model_name,
            temperature=settings.temperature,
        )

    prompt = HumanMessage(
        content=[
            {"type": "text", "text": inputs["prompt"]},
            {"type": "text", "text": inputs["format_instructions"]},
            {
                "type": "image_url",
                "image_url": {"url": inputs["image_url"]},
            },
        ]
    )
    logging.info(re.sub(r"(data:image/[^;]+;base64,)[^'\"]+", r"\1...", str(prompt.content)))

    response = model.invoke([prompt])

    return response.content


def get_image_informations(image_path: str) -> dict:
    """画像からテキスト情報を取得する"""
    parser = JsonOutputParser(pydantic_object=ImageInformation)
    format_instructions = parser.get_format_instructions()
    prompt = textwrap.dedent(
        """
            # 役割
            あなたは漫画の画像解析とデータ構造化に特化したエキスパートです。
            提供された漫画のコマ画像から、検索性に優れたテキスト情報と視覚的なタグ情報を抽出し、
            指定されたJSONスキーマに従って出力してください。
        """
    ).strip()

    load_image_chain = TransformChain(
        input_variables=["image_path"],
        output_variables=["image_url"],
        transform=load_image,
    )

    vision_chain = load_image_chain | image_model | parser
    return vision_chain.invoke(
        {
            "prompt": prompt,
            "format_instructions": format_instructions,
            "image_path": image_path,
        }
    )


def extract_filename_info(filename: str) -> dict:
    """ファイル名からidとpublishDateを抽出"""
    match = re.search(r"-(\d+)-\d{2}-(\d{14})\..+$", filename)
    if not match:
        return {"id": None, "publishDate": None}

    image_id = settings.x_url + match.group(1)
    embed = settings.twitter_url + match.group(1)
    publish_date_str = match.group(2)

    # JSTからUTCに変換し、ISO 8601形式にする
    jst = timezone(timedelta(hours=9))
    dt = datetime.strptime(publish_date_str, "%Y%m%d%H%M%S").replace(tzinfo=jst)
    publish_date_iso = dt.astimezone(timezone.utc).isoformat()

    return {"id": image_id, "embed": embed, "publishDate": publish_date_iso}


def process_images(image_dir: str, output_dir: str) -> None:
    """ディレクトリ内の画像を処理し、JSONに結果を保存"""
    image_dir_path, output_dir_path = Path(image_dir), Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    for image_file in sorted(image_dir_path.iterdir()):
        if image_file.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        logging.info(f"Image file: {image_file.name}")

        output_path = output_dir_path / image_file.with_suffix(".json").name

        # 既にJSONファイルが存在する場合はスキップ
        if output_path.exists():
            logging.info(f"Skipping {image_file.name}, output already exists.")
            continue

        info = {
            "id": None,
            "embed": None,
            "publishDate": None,
            "text": [],
            "reading": [],
            "tag": [],
            "series": [],
            "characters": [],
            "relatedLinks": [],
        }
        info.update(extract_filename_info(image_file.name))
        info.update(get_image_informations(str(image_file)))

        if "text" in info and isinstance(info["text"], list):
            info["text"] = [clean_text(t) for t in info["text"]]

        if "reading" in info and isinstance(info["reading"], list):
            info["reading"] = [clean_reading(r) for r in info["reading"]]

        logging.info(info)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=4)
            f.write("\n")


def main() -> None:
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(description="存在しない漫画の1コマbot(@noreal_koma)の漫画からテキストデータを抽出する")
    parser.add_argument("--ollama", action="store_true", help="ローカルのOllama APIを使用する")
    parser.add_argument("--model", type=str, help="使用するモデル名を指定する")
    args = parser.parse_args()

    if args.ollama:
        settings.use_ollama = True
        settings.model_name = args.model if args.model else settings.ollama_model_name
    elif args.model:
        settings.model_name = args.model

    process_images(settings.image_dir, settings.output_dir)


if __name__ == "__main__":
    main()
