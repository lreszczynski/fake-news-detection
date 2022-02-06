import glob
import json
import sys

import pandas as pd


def traverse_directories(location):
    df = pd.DataFrame(columns=['text', 'target'])
    rumours = glob.glob(location+'/**/rumours/*/source-twee*/*.json',
                        recursive=True)
    non_rumours = glob.glob(location+'/**/non-rumours/*/source-twee*/*.json',
                            recursive=True)

    for file in rumours:
        with open(file) as f:
            data = json.load(f)
            # print()
            df.loc[len(df.index)] = [data['text'], 1]

    for file in non_rumours:
        with open(file) as f:
            data = json.load(f)
            # print()
            df.loc[len(df.index)] = [data['text'], 0]

    return df


if (len(sys.argv) < 2):
    print("Uruchomienie programu: python convert.py sciezka_do_katalogu wynik.csv")
    exit(1)

input = sys.argv[1]
output = sys.argv[2]

df = traverse_directories(input)
df.to_csv(output, index=False)
