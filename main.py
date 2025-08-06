import base64
import json
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
                - 画像内のセリフや背景に書かれた文字などをテキストとして書き出します
                - 改行は省略し、1つの文字列としてまとめてください
                - 認識できる文字がない場合は空のリストを返します
            """
        ).strip()
    )

    reading: list[str] = Field(
        description=textwrap.dedent(
            """
                - textのよみがなをひらがなで提供します
                - textの要素1つにつき、1つの文字列が対応するリストです
                - よみがなが分からない場合は空の文字列を返します
            """
        ).strip()
    )

    tag: list[str] = Field(
        description=textwrap.dedent(
            """
                - 画像内に描かれている物をpixivのタグのような形式でリストアップします
                - できるだけ具体的な要素を抽出してください
                - 例として以下のようなタグを参考にしてください: 「獣人」「銃」「セーラー服」「眼鏡」「スーツ」
                - 一般的なタグは除外してください (例:「漫画」「キャラクター」「セリフ」)
                - 使用禁止タグ: アクション,アニメ,キャラクター,コミック,コメディ,シチュエーション,セリフ,パニック,会話,喜び,思考,恐怖,感情,感情,日常,汗,演技,漫画,緊張,表情,驚き
                - タグとして出力できるものがない場合は空のリストを返します
            """
        ).strip()
    )


def clean_text(text) -> str:
    """
    以下の処理を行う
    - 指定された文字を変換
    - 結合文字等を正規化
    """
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
    model = ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
    )

    msg = model.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": inputs["prompt"]},
                    {"type": "text", "text": inputs["format_instructions"]},
                    {
                        "type": "image_url",
                        "image_url": {"url": inputs["image_url"]},
                    },
                ]
            )
        ]
    )

    return msg.content


def get_image_informations(image_path: str) -> dict:
    """画像からテキスト情報を取得する"""
    parser = JsonOutputParser(pydantic_object=ImageInformation)
    format_instructions = parser.get_format_instructions()
    prompt = textwrap.dedent(
        """
            画像内のセリフや背景に書かれた文字などをすべて、正確にテキストとして書き出してください。
            要するにOCRです。
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
    publish_date_str = match.group(2)

    # JSTからUTCに変換し、ISO 8601形式にする
    jst = timezone(timedelta(hours=9))
    dt = datetime.strptime(publish_date_str, "%Y%m%d%H%M%S").replace(tzinfo=jst)
    publish_date_iso = dt.astimezone(timezone.utc).isoformat()

    return {"id": image_id, "publishDate": publish_date_iso}


def process_images(image_dir: str, output_dir: str) -> None:
    """ディレクトリ内の画像を処理し、JSONに結果を保存"""
    image_dir_path, output_dir_path = Path(image_dir), Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    for image_file in sorted(image_dir_path.iterdir()):
        if image_file.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        print(f"Image file: {image_file.name}")

        output_path = output_dir_path / image_file.with_suffix(".json").name

        # 既にJSONファイルが存在する場合はスキップ
        if output_path.exists():
            print(f"Skipping {image_file.name}, output already exists.")
            continue

        info = {
            "id": None,
            "publishDate": None,
            "text": [],
            "reading": [],
            "tag": [],
            "series": [],
            "characters": [],
            "suzuri": [],
            "lineSticker": [],
            "instagram": "",
            "QRCode": [],
        }
        info.update(extract_filename_info(image_file.name))
        info.update(get_image_informations(str(image_file)))

        if "text" in info and isinstance(info["text"], list):
            info["text"] = [clean_text(t) for t in info["text"]]

        if "reading" in info and isinstance(info["reading"], list):
            info["reading"] = [clean_reading(r) for r in info["reading"]]

        print(info)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=4)
            f.write("\n")


def main() -> None:
    process_images(settings.image_dir, settings.output_dir)


if __name__ == "__main__":
    main()
