
from __future__ import print_function

from fileinput import filename

import matplotlib.pyplot as plt
import codecs
import TextRank4Sentence
import numpy as np
import csv
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
def plot_rouge_line(rouge_scores, title="ROUGE Scores", save_path=None):

    """绘制单篇文本的ROUGE指标折线图"""

    metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    scores = [rouge_scores['rouge1_f1'], rouge_scores['rouge2_f1'], rouge_scores['rougeL_f1']]
    print(123)
    plt.figure(figsize=(8, 5))
    plt.plot(metrics, scores, marker='o', linestyle='-', linewidth=2, color='royalblue')
    print(123)
    # 添加数据标签
    for x, y in zip(metrics, scores):
        plt.text(x, y, f'{y:.2f}', ha='center', va='bottom')
    print(123)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.title(title, fontsize=14)
    plt.ylim(0, 1)
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    print(123)
    plt.savefig(save_path)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()

text = codecs.open('./doc/01.txt', 'r', 'utf-8').read()
text_=[]
result_file=r"C:\Users\30505\Desktop\textRank\zhaiyao\NLP_2\data\predict_chouqu.tsv"
text_zhai=[]
filename=r'C:\Users\30505\Desktop\textRank\zhaiyao\NLP_1\doc\01.txt'
# with open(filename, encoding='utf-8') as f:
    # for l in f.readlines():
    #     cur = l.strip().split('\t')
    #     if len(cur) == 2:
    #         title, content = cur[0], cur[1]
    #         text_.append(content)
    #         text_zhai.append(title)
num=4
tr4s = TextRank4Sentence.TextRank4Sentence()
j=1
method_scores={}
reference_text = text  # 参考摘要
i=0
method='cosine'
#for text,reference_text in zip(text_,text_zhai):
for method in ['cosine','yuan']:
    #,'sbert','hybrid','wordvec'
    i+=1
    tr4s.analyze(text=text, lower=True, source = 'all_filters',similarity_method=method)
    for st in tr4s.sentences:
        print(type(st), st)

    print(20*'*')
    with open(result_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        gen=','.join(item.sentence for item in tr4s.get_key_sentences(num=num))
        writer.writerow([gen])
        # for item in tr4s.get_key_sentences(num=num):
        #     gen = [item_gen.replace('\t', '').replace('\n','') for item_gen in item.sentence]
        #     gen=''.join(gen)
        #     gen=gen.replace('\t','')
        #     #writer.writerows(zip(item.sentence, text_))
        #     print(gen+'服死了')
        #     writer.writerow([gen])
        #     print(item.weight, item.sentence, type(item.sentence))

    summary = ' '.join([item.sentence for item in tr4s.get_key_sentences(num=num)])
    print("wwwwwww")
    for j in summary:
        print(j.replace('\n',''))
    print("wwwwwww")
    scores = tr4s.evaluate_summary(reference_text)
    print("ROUGE Scores:", scores)
    import os
    # test_1.py中添加绘图代码
    output_dir = r'C:\Users\30505\Desktop\textRank\zhaiyao\NLP_1\picture'
    #os.makedirs(output_dir, exist_ok=True)
    tr4s.plot_sentence_scores(save_path=r'C:\Users\30505\Desktop\textRank\zhaiyao\NLP_1\picture\sentence_scores{}.png'.format(i))  # 保存为图片
    #j=j+1

    #method_scores[method]
    print(123)
    plot_rouge_line(scores, title=f"ROUGE ({method})",save_path=f'picture/ROUGE ({method})')