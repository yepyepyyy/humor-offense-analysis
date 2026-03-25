import csv
import json
import os

from IPython.core.debugger import prompt
from openai import OpenAI
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = os.getenv("BASE_URL")
#model=os.getenv("MODEL")
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)
#completion = client.chat.completions.create(model="qwen2.5-3b-instruct",messages=[{'role': 'user', 'content': '你是谁？'}])

#print(completion.choices[0].message.content)



def convert_csv(in_path, out_path, text_col, label_col):
    """
    把旧 CSV 转成新表头：
    id,text,is_humor,humor_rating,humor_controversy,offense_rating
    其中 humor_rating/controversy/offense 先留空（或你可改成默认值）
    """
    with open(in_path, "r", encoding="utf-8") as f_in, \
         open(out_path, "w", encoding="utf-8-sig", newline="") as f_out:

        reader = csv.DictReader(f_in)
        fieldnames = ["id", "text", "is_humor", "humor_rating", "humor_controversy", "offense_rating"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()



        for i, row in enumerate(reader):
            rid = (row.get("id", "") or "").strip()
            if rid == "":
                rid = str(i)

            new_row = {
                "id": rid,
                "text": row.get(text_col, ""),
                "is_humor": row.get(label_col, ""),
                "humor_rating": "",           # 先空着
                "humor_controversy": "",      # 先空着
                "offense_rating": "",         # 先空着
            }
            writer.writerow(new_row)

    print("Saved:", out_path)
def score_with_qwen(text:str):
    """
    返回 dict:
    {"humor_rating": float(0-5), "humor_controversy": float(0-5 or 0-1), "offense_rating": float(0-5)}
    你可以按你想要的范围改提示词。
    """
    prompt=f"""You are annotating humor data. Given the text, output STRICT JSON only with keys:
humor_rating (0-5 float),
humor_controversy (0-5 float),
offense_rating (0-5 float).
Text：{text}"""
    prompt=prompt.strip()
    resp=client.chat.completions.create(
        model="qwen2.5-3b-instruct",
        messages=[{"role":"user","content":prompt}],
        temperature=0)
    s=resp.choices[0].message.content.strip()
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        start = s.find("{")
        end = s.rfind("}")
        data = json.loads(s[start:end + 1])
    return {
        "humor_rating": data.get("humor_rating", ""),
        "humor_controversy": data.get("humor_controversy", ""),
        "offense_rating": data.get("offense_rating", ""),
    }
def convert_and_score(in_path, out_path,text_col, label_col, limit=None,start_line=0):
    with open(in_path, "r", encoding="utf-8") as f_in, \
         open(out_path, "w", encoding="utf-8-sig", newline="") as f_out:

        reader = csv.DictReader(f_in)
        fieldnames = ["id", "text", "is_humor", "humor_rating", "humor_controversy", "offense_rating"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(reader):
            if i<start_line:
                continue
            if limit is not None and (i-start_line) >= limit:
                break

            text = row.get(text_col, "")
            scores = score_with_qwen(text) if text else {"humor_rating":"", "humor_controversy":"", "offense_rating":""}

            rid = (row.get("id", "") or "").strip()
            if rid == "":
                rid = str(i)

            new_row = {
                #"id": row.get("id", ""),
                "id": rid,
                "text": text,
                "is_humor": row.get(label_col, ""),
                **scores
            }
            writer.writerow(new_row)

    print("Saved:", out_path)


with open('./English/train.csv','r',encoding='utf-8') as Eng_data:
    content=Eng_data.read()
    #print(content)

with open('./Chinese/task1/task1_train.csv','r',encoding='utf-8') as Eng_data:
    content=Eng_data.read()
   # print(content)

# 中文：如果表头不同（比如 joke/label 叫别的），把 text_col / label_col 改成真实列名
#convert_csv("./Chinese/task2/task2_test_with_label.csv", "./Chinese/task2/task2_test_with_label_converted.csv",text_col="joke", label_col="label")
#convert_csv("./Chinese/task1/task1_test_with_label.csv", "./Chinese/task1/task1_test_with_label_converted.csv",text_col="joke", label_col="label")
# 补充分数
convert_and_score("./Chinese/task1/task1_train.csv", "Chinese/task1/task1_train_scored5.csv", text_col="text", label_col="is_humor", limit=2000, start_line=8743)
#convert_and_score("./Chinese/task/task1_test_with_label.csv", "Chinese/task1/task1_test_with_label_scored.csv", text_col="text", label_col="is_humor" )
