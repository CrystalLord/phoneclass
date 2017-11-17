def one_hotify(dic):
    """Convert a dictionary's keys to a one hot vector"""
    new_dic = {}
    sym_count = max(tuple(dic.values()))+1
    for k in dic.keys():
        one_hot = [0]*sym_count
        one_hot[dic[k]] = 1
        new_dic[k] = one_hot
    return new_dic
