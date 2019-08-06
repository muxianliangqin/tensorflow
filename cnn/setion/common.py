import collections


class Section(collections.namedtuple('Section', ['scope', 'fn', 'args'])):
    '''
    :param scope: 变量域
    :param fn: 执行方法
    :param fn: 执行参数
    '''

