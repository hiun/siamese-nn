import nltk
tokenizer = nltk.tokenize.TweetTokenizer(strip_handles=True, reduce_len=True)

# construct char and word - level dics
def _get_char_idx_mapping(content_list):
    uniq_char_list = set()  # use SET to make fast traversal
    uniq_char_list.add(' ') # make SPACE as a first index; check code redundancy
    for content in content_list:
        for char in content:
            if char not in uniq_char_list:
                uniq_char_list.add(char)

    uniq_char_list = sorted(list(uniq_char_list))
    uniq_chars_index = dict([(char, i) for i, char in enumerate(uniq_char_list)])
    return uniq_char_list, uniq_chars_index

def _get_word_idx_mapping(content_list):
    uniq_word_list = set()
    for content in content_list:
        word_list = nltk.word_tokenize(content)
        for word in word_list:
            if word not in uniq_word_list:
                uniq_word_list.add(word)
    uniq_word_list = sorted(list(uniq_word_list))
    uniq_words_index = dict([(word, i) for i, word in enumerate(uniq_word_list)])
    return uniq_word_list, uniq_words_index
            
def byte_lst2char(content_list):
    return list(map(lambda v: (
        v.decode("utf8")
    ), content_list))

def limit_length_to_N(content_list, N):
    return list(map(lambda v: (
        v[:N]
    ), content_list))

def limit_length_to_N_backword(content_list, N):
    return list(map(lambda v: (
        v[N:]
    ), content_list))

def pad_zero_to_N(lst, N):
    return list(map(lambda item: (
        item + ([0] * (N - len(item))) # TODO: check setting "0" is the right way to implement the semantics of emptiness.
    ), lst))

def str2idx(str, uniq_item_index):
    result_idx_list = []
    for item in str:
        result_idx_list.append(uniq_item_index[item])
    return result_idx_list

def str_lst2idx(lst, uniq_item_index):
    return list(map(lambda str: (
        str2idx(str, uniq_item_index)
    ), lst))

def preproc_char(_char_lst_q1, uniq_chars_index, MAX_CHAR_LENGTH):    
    content_chars_lst = []
    for content in _char_lst_q1:
        try:
            _content_chars_lst = list(content)
        except Exception as e:
            raise Exception(e)
        content_chars_lst.append(_content_chars_lst)

    content_chars_lst = limit_length_to_N(content_chars_lst, MAX_CHAR_LENGTH)
        
    _idx_list_q1 = str_lst2idx(content_chars_lst, uniq_chars_index)

    _padded_idx_list_q1 = pad_zero_to_N(_idx_list_q1, MAX_CHAR_LENGTH)
    
    return _padded_idx_list_q1

def preproc_word(content_lst, uniq_words_index, MAX_CHAR_LENGTH):
    contents_word_lst = []
    for content in content_lst: 
        content_word_lst = nltk.word_tokenize(content)
        contents_word_lst.append(content_word_lst)

    pruned_word_lst = limit_length_to_N(contents_word_lst, MAX_CHAR_LENGTH)

    word_idx_lst = str_lst2idx(pruned_word_lst, uniq_words_index)

    padded_word_idx_lst = pad_zero_to_N(word_idx_lst, MAX_CHAR_LENGTH)
    
    return padded_word_idx_lst

def _get_char_length_list(content_list, MAX_CHAR_LENGTH):
    length_list = []
    for content in content_list:
        if type(content) is not str: # TODO: check code redundancy
            print('malformed content')
            print(content)
        length_list.append(min(len(content), MAX_CHAR_LENGTH))
    return length_list

def _get_word_length_list(content_list, MAX_WORD_LENGTH):
    length_list = []
    for content in content_list:
        length_list.append(min(len(content.split(' ')), MAX_WORD_LENGTH))
    return length_list

def limit_length_to_N_backword(content_list, N):
    return list(map(lambda v: (
        v[-N:]
    ), content_list))