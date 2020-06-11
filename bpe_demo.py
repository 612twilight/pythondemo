import re


def process_raw_words(words, endtag='-'):
    '''把英文单词分成每个字母，然后每个单词最后加上结束符，这里结束符-'''
    vocabs = {}
    for word, count in words.items():
        # 加上空格
        word = re.sub(r'([a-zA-Z])', r' \1', word)
        word += ' ' + endtag
        vocabs[word] = count
    return vocabs


def get_symbol_pairs(vocabs):
    ''' 组合连续 相邻的两个字母，并统计出现的频率
    Args:
        vocabs: 单词dict，(word, count)单词的出现频率。单词已经被分割成字母
    Returns:
        pairs: dict,{(符号1, 符号2):count}
    '''
    # pairs = collections.defaultdict(int)
    pairs = dict()
    for word, freq in vocabs.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            p = (symbols[i], symbols[i + 1])
            pairs[p] = pairs.get(p, 0) + freq  # 累加相同字母组合的频率
    return pairs


def merge_symbols(symbol_pair, vocabs):
    '''把vocabs中的所有单词中的'a b'字符串用'ab'替換
    Args:
        symbol_pair: (a, b) 两个元素
        vocabs: dict，(word：count)。其中word使用空格分割
    Returns:
        vocabs_new: 替换'a b'为'ab'后的新词表
    '''
    vocabs_new = {}
    raw = ' '.join(symbol_pair)
    merged = ''.join(symbol_pair)
    # 把非字母和数字做转义
    bigram = re.escape(raw)
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word, count in vocabs.items():
        word_new = p.sub(merged, word)
        vocabs_new[word_new] = count
    return vocabs_new


def main():
    raw_words = {"low": 5, "newest": 6, "widest": 3}  # lower
    vocabs = process_raw_words(raw_words)

    num_merges = 10
    print(vocabs)
    for i in range(num_merges):
        pairs = get_symbol_pairs(vocabs)
        print(pairs)
        # 选择出现频率最高的pair
        symbol_pair = max(pairs, key=pairs.get)
        vocabs = merge_symbols(symbol_pair, vocabs)
        print(vocabs)
        print('-' * 30)
    print(vocabs)


if __name__ == '__main__':
    main()