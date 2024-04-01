import csv
import openai
import argparse
import pandas as pd

openai.api_key=""

def translate(client, text):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "請將下列台文文章轉換成繁體中文"},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str)
    parser.add_argument("-o", "--output", type=str)
    args = parser.parse_args()
    
    client = openai.OpenAI()
    df = pd.read_csv(args.file)
    for i in range(len(df)):
        df.iloc[i]["text"] = translate(client, df.iloc[i]["text"])
    df.to_csv(args.output, index=None)

if __name__ == "__main__":
    main()