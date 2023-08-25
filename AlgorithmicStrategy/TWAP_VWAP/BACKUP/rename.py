from pathlib import Path


def file_rename(replaced_word='', new_word='', fp=None):
    '''
    批量修改文件或文件夹名称，只需给出主目录，会扫描主目录下的所有文件及文件夹
    :params
        :replaced_word 被替换的原始字符串
        :new_word 新的字符串
        :fp 主文件路径
    '''

    # 替换参数验证
    # if not replaced_word or not new_word:
    #     raise TypeError('请确认输入了被替换字符串和替换字符串')

    # 主动检查路径参数问题，抛出脑瘫问题
    if not fp:
        raise TypeError('参数错误,缺失参数fp:文件路径')

    # if not fp.is_dir() or fp.is_file():
    #     raise TypeError('找不到文件目录,请确认')

    # 如果传入路径参数为字符串，则转为Path类型
    fp = Path(fp) if isinstance(fp, str) else fp

    # 提取主路径下的文件夹列表，并格式化路径问字符串，以便后续的替换操作
    # replace是没用的，但是已经写了，想修改吧，注释写了这么多了，算了，不改了，就这样吧
    dirs = [x.__str__().replace('\\', '/') for x in fp.iterdir()] if fp.is_dir() else []

    # 计数器，没啥鸟用
    total = 0
    for path in dirs:
        # 被替换字符串存在于path路径中，直接替换
        if replaced_word in path:

            new_nanme = path.replace(replaced_word, new_word)
            new_fp = fp / new_nanme
            target = fp / path
            target.rename(new_fp)
            total += 1
            print(f'success: old:{path} new:{new_fp}')
            print(f'rename nums:{total}')

            # 递归检查文件夹
            if Path(new_fp).is_dir():
                file_rename(replaced_word, new_word, fp=new_fp)
        # 当前路径中没有被替换的字符串，继续递归向下检查
        else:
            file_rename(replaced_word, new_word, Path(path))


# 调用
if __name__ == '__main__':
    fp = Path(r'./')
    # fp = 'E:\online project\'
    replaced_word = 'little'
    new_word = ''
    file_rename(replaced_word=replaced_word, new_word=new_word, fp=fp)
