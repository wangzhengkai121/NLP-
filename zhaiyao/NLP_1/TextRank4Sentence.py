import networkx as nx
import numpy as np
from rouge_score import rouge_scorer
import util
import Segmentation
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
class TextRank4Sentence(object):

    def __init__(self, stop_words_file=None,  # 停止词文件路径
                 allow_speech_tags=util.allow_speech_tags,  # 词的词性e.g名词、动词之类的
                 delimiters=util.sentence_delimiters):  # delimiters       --  默认值是`?!;？！。；…\n`，用来将文本拆分为句子。
        """
        Keyword arguments:
        stop_words_file  --  str，停止词文件路径，若不是str则是使用默认停止词文件


        Object Var:
        self.sentences               --  由句子组成的列表。
        self.words_no_filter         --  对sentences中每个句子分词而得到的两级列表。
        self.words_no_stop_words     --  去掉words_no_filter中的停止词而得到的两级列表。
        self.words_all_filters       --  保留words_no_stop_words中指定词性的单词而得到的两级列表。
        """
        self.seg = Segmentation.Segmentation(stop_words_file=stop_words_file,
                                allow_speech_tags=allow_speech_tags,
                                delimiters=delimiters)

        self.sentences = None
        self.words_no_filter = None  # 2维列表
        self.words_no_stop_words = None
        self.words_all_filters = None

        self.key_sentences = None

    # def analyze(self, text, lower=False,
    #             source='no_stop_words',
    #             sim_func=util.get_similarity,
    #             pagerank_config={'alpha': 0.85, }):
    def analyze(self, text, lower=False, source='no_stop_words',sim_func=util.get_similarity, pagerank_config={'alpha': 0.85, },similarity_method='cosine'):
        """
        Keyword arguments:
        text                 --  文本内容，字符串。
        lower                --  是否将文本转换为小写。默认为False。
        source               --  选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来生成句子之间的相似度。
                                 默认值为`'all_filters'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'`。
        sim_func             --  指定计算句子相似度的函数。
        """

        self.key_sentences = []

        result = self.seg.segment(text=text, lower=lower)
        self.sentences = result.sentences
        self.words_no_filter = result.words_no_filter
        self.words_no_stop_words = result.words_no_stop_words
        self.words_all_filters = result.words_all_filters

        options = ['no_filter', 'no_stop_words', 'all_filters']
        if source in options:
            _source = result['words_' + source]
        else:
            _source = result['words_no_stop_words']

        self.key_sentences = util.sort_sentences(sentences=self.sentences,  # 将句子按照关键程度从大到小排序
                                                 words=_source,
                                                 sim_func=sim_func,
                                                 similarity_method=similarity_method,
                                                 #sim_func=lambda x, y: sim_func(x, y,method=similarity_method),
                                                 pagerank_config=pagerank_config)

    def get_key_sentences(self, num=6, sentence_min_len=6):
        """获取最重要的num个长度大于等于sentence_min_len的句子用来生成摘要。

        Return:
        多个句子组成的列表。
        """
        result = []
        count = 0
        for item in self.key_sentences:
            if count >= num:
                break
            if len(item['sentence']) >= sentence_min_len:
                result.append(item)
                count += 1
        return result

    def evaluate_summary(self, reference_text):
        """计算ROUGE指标（参考文本为原文）"""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        # 生成摘要
        summary = ' '.join([item.sentence for item in self.get_key_sentences()])
        # 计算指标
        scores = scorer.score(reference_text, summary)
        return {
            "rouge1_f1": scores["rouge1"].fmeasure,
            "rouge2_f1": scores["rouge2"].fmeasure,
            "rougeL_f1": scores["rougeL"].fmeasure
        }

    def plot_sentence_scores(self, save_path=None):
        """绘制句子得分分布柱状图"""
        if not self.key_sentences:
            raise ValueError("请先调用analyze方法生成关键句子")
        scores = [item.weight for item in self.key_sentences]
        sentences = [item.sentence for item in self.key_sentences]
        plt.figure(figsize=(10, 6))
        plt.bar(range(1,len(sentences)+1), scores, color='skyblue')
        plt.xlabel('句子', fontsize=12)
        plt.ylabel('TextRank得分', fontsize=12)
        plt.title('句子重要性得分分布', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

if __name__ == '__main__':
    pass