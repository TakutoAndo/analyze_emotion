import pandas as pd
import numpy as np
import os
import glob
import pathlib
import re
import janome
import jaconv
import sys

#極性辞書までのpath
p_dic = pathlib.Path('dic')

for i in p_dic.glob('*.txt'):
    with open (i, 'r', encoding= 'cp932') as f:
        x = [ii.replace('\n', '').split(':') for ii in f.readlines()]

#行にラベル付けしてリスト化
posi_nega_df = pd.DataFrame(x, columns = ['基本形', '読み', '品詞', 'スコア'])
#print(posi_nega_df)

#jaconvで読みをカタカナに変換
posi_nega_df['読み'] = posi_nega_df['読み'].apply(lambda x: jaconv.hira2kata(x))

#読みや品詞が同じで、異なるスコアが割り当てられていたものは重複を削除
#.duplicated(): 重複要素をtrueで返す
# + ~posi_nega_df[]: ビット反転した配列のindex
# -> falseが返ってきた要素(重複なしのposi_nega_df[])のindex
posi_nega_df = posi_nega_df[~posi_nega_df[['基本形', '読み', '品詞']].duplicated()]
#print(posi_nega_df)

p_text = pathlib.Path('text')

article_list = []

#フォルダ内のテキストファイルをサーチ
for p in p_text.glob('**/*.txt'):
    #第二階層フォルダ名がニュースサイトの名前になっているので、それを取得
    file_info = str(p).split('/')
    #CHANGES.txt, README.txtは無視
    if(len(file_info) == 3):
        media = str(p).split('/')[1]
        file_name = str(p).split('/')[2]

        if file_name != 'LICENSE.txt':
            #テキストファイル読み込み
            with open(p, 'r') as f:
                #テキストファイルの中身を1行ずつ読み込み、リスト形式で格納
                article = f.readlines()
                #本文内、改行文字と空白を削除するように置換
                article = [re.sub(r'[\n \u3000]', '', i) for i in article]
            #ニュースサイト名、記事URL、日付、記事タイトル、本文の並びでリスト化
            #''.join(article[3:]) -> 本文も一行ずつarticleにはいってるから、3番目以降(本文)を結合していく-> 一個の要素に
            article_list.append([media, article[0], article[1], article[2], ''.join(article[3:])])
        else:
            continue

article_df = pd.DataFrame(article_list)

article_df.head()

#ここから形態素分析

#topic-news内記事を分析
news_df = article_df[article_df[0] == 'topic-news'].reset_index(drop = True)
#print(news_df)

from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.charfilter import *

t = Tokenizer()
char_filters = [UnicodeNormalizeCharFilter()]
analyzer = Analyzer(char_filters=char_filters, tokenizer=t)

word_lists = []
for i, row in news_df.iterrows():
    for t in analyzer.analyze(row[4]):
        #形態素
        surf = t.surface
        #基本形
        base = t.base_form
        #品詞
        pos = t.part_of_speech
        #読み
        reading = t.reading

        word_lists.append([i, surf, base, pos, reading])

word_df = pd.DataFrame(word_lists, columns= ['ニュースNo.', '単語', '基本形', '品詞', '読み'])
word_df['品詞'] = word_df['品詞'].apply(lambda x : x.split(',')[0])

#基本形、品詞、読みが一致する箇所(onで明示できる)をマージ
score_result = pd.merge(word_df, posi_nega_df, on = ['基本形', '品詞', '読み'], how = 'left')

result = []
for i in range(len(score_result['ニュースNo.'].unique())):
    temp_df = score_result[score_result['ニュースNo.'] == i]
    text = ''.join(list(temp_df['単語']))
    score = temp_df['スコア'].astype(float).sum()
    #スコアをスコアが付与されている単語数で割った値
    score_r = score/temp_df['スコア'].astype(float).count()
    result.append([i, text, score, score_r])

final = pd.DataFrame(result, columns= ['ニュースNo.', 'テキスト', '累計スコア', '標準化スコア']).sort_values(by = '標準化スコア').reset_index(drop = True)
#print(final)

def writeTextFile(file_name, text):
    f = open(file_name, 'w')
    f.write(text)
    f.close()

posi_best = final[final['ニュースNo.'] == 510]
nega_best = final[final['ニュースNo.'] == 57]
pd.set_option("display.max_colwidth", 1000)
posi_best = str(posi_best['テキスト'])
nega_best = str(nega_best['テキスト'])
writeTextFile('nega_best.txt', nega_best)
writeTextFile('posi_best.txt', posi_best)
df_sample1 = posi_nega_df.head(20)
df_sample2 = posi_nega_df.tail(20)
df_sample = df_sample1.append(df_sample2)
writeTextFile('posi_nega_df.txt', str(df_sample))