import csv
import openai
import argparse
import pandas as pd

openai.api_key="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"

def translate(client, text):
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": 
             "你是一位台語老師，要將下列台文的句子轉換成繁體中文，如同以下兩個範例\n\
             1.\n \
             今仔日油價一公升起五角。\n \
             今天油價每公升上漲五角。\n \
             2.\n \
             淡水明仔載的溫度二十八到三十度，落雨機率四十％。\n \
             淡水今天的溫度二十八到三十度，降雨機率四十％。\n \
             ，請不要更改原本的語意或加入其他不相關的內容，並請直接回答答案。"},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str)
    parser.add_argument("-o", "--output", type=str)
    args = parser.parse_args()
    
    client = openai.OpenAI(api_key=openai.api_key)
    df = pd.read_csv(args.file)
    from tqdm import tqdm
    for i in tqdm(range(len(df))):
        df.loc[i, "text"] = translate(client, df.loc[i, "text"])
    df.to_csv(args.output, index=None)

if __name__ == "__main__":
    main()
