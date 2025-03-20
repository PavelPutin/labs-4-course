class Node:
    # Статическая переменная для нумерации узлов
    __num__ = -1

    def __init__(self, parent_key, out_edges, suffix_link=None):
        # Ключ родительского узла
        self.parent_key = parent_key
        # Словарь исходящих ребер узла
        self.out_edges = out_edges
        # Суффиксная ссылка
        self.suffix_link = suffix_link
        # Увеличиваем счетчик узлов и присваиваем уникальный идентификатор
        Node.__num__ += 1
        self.id = Node.__num__

    def get_out_edges(self):
        return self.out_edges

    def set_out_edge(self, key, anode, label_start, label_end, bnode):
        # Добавление исходящего ребра к узлу
        self.out_edges = self.out_edges or {}
        self.out_edges[key] = (anode, label_start, label_end, bnode)

    def get_out_edge(self, key):
        return self.out_edges.get(key, None)

    def get_parent_key(self):
        return self.parent_key

    def set_parent_key(self, parent_key):
        self.parent_key = parent_key

    def get_suffix_link(self):
        return self.suffix_link

    def set_suffix_link(self, node):
        self.suffix_link = node

    def get_id(self):
        return self.id

    @staticmethod
    def __draw__(root_node, chars, v, ed='#'):
        # Рекурсивный метод для отрисовки дерева
        # Разделение исходящих ребер на те, которые ведут к узлам без детей, и те, которые ведут к узлам с детьми
        edges = root_node.get_out_edges().items()
        no_gc, has_gc = [], []
        for edge in edges:
            if edge[1][3].get_out_edges() is None:
                no_gc.append(edge)
            else:
                has_gc.append(edge)

        # Порядок отрисовки ребер: сначала ребра к узлам с детьми, затем ребра к листьям
        gc = has_gc + no_gc if v == 0 else no_gc + has_gc
        maxlen = len(chars) + 6

        for k, (parent, s, t, node) in gc:
            # Если конец ребра не указан, устанавливаем его как длину строки или специальный символ
            if t == '#':
                t = len(chars) if ed == '#' else ed
            linkid = ''
            # Добавляем суффиксную ссылку, если она существует
            if node.get_suffix_link() is not None:
                linkid = '->' + str(node.get_suffix_link().get_id())

            # Отрисовка ребра и рекурсивный вызов для его потомков
            if v == 0:
                print(f'{" " * maxlen * v}|\n'
                      f'{" " * maxlen * v}|{" " * 3}{chars[s:t + 1]}\n'
                      f'+{" " * maxlen * v}-{"-" * (maxlen - 1)}● ({str(node.get_id()) + linkid})')
            else:
                print(f'|{" " * maxlen * v}|\n'
                      f'|{" " * maxlen * v}|{" " * 3}{chars[s:t + 1]}\n'
                      f'|{" " * maxlen * v}+{"-" * (maxlen - 1)}● ({str(node.get_id()) + linkid})')
            if node.get_out_edges():
                Node.__draw__(node, chars, v + 1, ed)

    @staticmethod
    def draw(root, chars, ed='#'):
        # Запуск отрисовки дерева, начиная с корневого узла
        print(f'\n{chars}\n● (0)')
        Node.__draw__(root, chars, 0, ed)


def build(chars, regularize=False):
    """
    Построение дерева суффиксов из строки chars
    """
    root = Node(None, None, None)  # Создание корневого узла
    act_node = root  # Текущий узел
    act_key = ''  # Текущий ключ (первый символ ребра)
    act_len = 0  # Длина текущего ребра
    remainder = 0  # Остаток строки для обработки
    ind = 0  # Индекс обрабатываемого символа

    while ind < len(chars):
        ch = chars[ind]
        if remainder == 0:
            # Если остаток равен 0, проверяем, существует ли исходящее ребро с символом ch
            if act_node.get_out_edges() and ch in act_node.get_out_edges():
                act_key = ch
                act_len = 1
                remainder = 1
                anode, start, end, bnode = act_node.get_out_edge(act_key)
                if end == '#':
                    end = ind
                if end - start + 1 == act_len:
                    act_node = bnode
                    act_key = ''
                    act_len = 0
            else:
                # Если ребра не существует, создаем новый лист
                leaf = Node(None, None, None)
                act_node.set_out_edge(chars[ind], act_node, ind, '#', leaf)
                leaf.set_parent_key((act_node, chars[ind]))
        else:
            if not act_key and not act_len:
                # Если мы на узле, проверяем, существует ли исходящее ребро с символом ch
                if ch in act_node.get_out_edges():
                    act_key = ch
                    act_len = 1
                    remainder += 1
                else:
                    # Если ребра не существует, разворачиваем дерево
                    remainder += 1
                    remainder, act_node, act_key, act_len = unfold(root, chars, ind, remainder, act_node, act_key,
                                                                   act_len)
            else:  # compare on edge
                # Если мы на ребре, проверяем, совпадают ли символы на ребре и в остатке строки
                out_edge = act_node.get_out_edge(act_key)
                if out_edge:
                    anode, start, end, bnode = out_edge
                    if end == '#':
                        end = ind
                    compare_pos = start + act_len
                    if chars[compare_pos] != ch:
                        # Если символы не совпадают, разворачиваем дерево
                        remainder += 1
                        remainder, act_node, act_key, act_len = unfold(root, chars, ind, remainder, act_node, act_key,
                                                                       act_len)
                    else:
                        if compare_pos < end:  # on edge
                            # Если мы все еще на ребре, увеличиваем длину ребра
                            act_len += 1
                            remainder += 1
                        else:
                            # Если мы достигли конца ребра, переходим к следующему узлу
                            remainder += 1
                            act_node = bnode
                            if compare_pos == end:
                                act_len = 0
                                act_key = ''
                            else:
                                act_len = 1
                                act_key = ch
        ind += 1
        if ind == len(chars) and remainder > 0:
            # Если мы обработали всю строку, но остаток не равен 0, добавляем специальный символ
            if regularize:
                chars += '$'
    return root, chars


def unfold(root, chars, ind, remainder, act_node, act_key, act_len):
    """
    Вспомогательная рекурсивная функция для разворачивания дерева
    """
    pre_node = None
    while remainder > 0:
        remains = chars[ind - remainder + 1:ind + 1]
        act_len_re = len(remains) - 1 - act_len
        act_node, act_key, act_len, act_len_re = hop(ind, act_node, act_key, act_len, remains, act_len_re)
        lost, act_node, act_key, act_len, act_len_re = step(chars, ind, act_node, act_key, act_len, remains, act_len_re)
        if lost:
            # Если мы потеряли путь в дереве, устанавливаем суффиксную ссылку и выходим из функции
            if act_len == 1 and pre_node is not None and act_node is not root:
                pre_node.set_suffix_link(act_node)
            return remainder, act_node, act_key, act_len
        if not act_len:
            # Если мы на узле и символ не совпадает с ребром, создаем новый лист
            if remains[act_len_re] not in act_node.get_out_edges():
                leaf = Node(None, None, None)
                act_node.set_out_edge(remains[act_len_re], act_node, ind, '#', leaf)
                leaf.set_parent_key((act_node, remains[act_len_re]))
        else:
            # Если мы на ребре, проверяем, совпадают ли символы
            out_edge = act_node.get_out_edge(act_key)
            if out_edge:
                anode, start, end, bnode = out_edge
                if remains[act_len_re + act_len] != chars[start + act_len]:
                    # Если символы не совпадают, разбиваем ребро
                    new_node = Node(None, None, None)
                    act_node.set_out_edge(act_key, act_node, start, start + act_len - 1, new_node)
                    new_node.set_parent_key((act_node, act_key))
                    new_node.set_out_edge(chars[start + act_len], new_node, start + act_len, end, bnode)
                    leaf = Node(None, None, None)
                    new_node.set_out_edge(chars[ind], new_node, ind, '#', leaf)
                    leaf.set_parent_key((new_node, chars[ind]))
                else:
                    return remainder, act_node, act_key, act_len
        # Устанавливаем суффиксную ссылку для нового узла
        if pre_node and 'leaf' in locals() and leaf.get_parent_key()[0] is not root:
            pre_node.set_suffix_link(leaf.get_parent_key()[0])
        if 'leaf' in locals() and leaf.get_parent_key()[0] is not root:
            pre_node = leaf.get_parent_key()[0]
        # Переходим к следующему узлу по суффиксной ссылке или к корневому узлу
        if act_node == root and remainder > 1:
            act_key = remains[1]
            act_len -= 1
        if act_node.get_suffix_link() is not None:
            act_node = act_node.get_suffix_link()
        else:
            act_node = root
        remainder -= 1
    return remainder, act_node, act_key, act_len


def step(chars, ind, act_node, act_key, act_len, remains, ind_remainder):
    """
    Вспомогательная функция для переходов по ребрам и узлам дерева
    """
    rem_label = remains[ind_remainder:]
    if act_len > 0:
        # Если мы на ребре, проверяем, совпадает ли остаток строки с меткой ребра
        out_edge = act_node.get_out_edge(act_key)
        if out_edge:
            anode, start, end, bnode = out_edge
            if end == '#':
                end = ind
            edge_label = chars[start:end + 1]
            if edge_label.startswith(rem_label):
                act_len = len(rem_label)
                act_key = rem_label[0]
                return True, act_node, act_key, act_len, ind_remainder
    else:
        # Если мы на узле, проверяем, существует ли исходящее ребро с символом из остатка строки
        if ind_remainder < len(remains) and remains[ind_remainder] in act_node.get_out_edges():
            out_edge = act_node.get_out_edge(remains[ind_remainder])
            if out_edge:
                anode, start, end, bnode = out_edge
                if end == '#':
                    end = ind
                edge_label = chars[start:end + 1]
                if edge_label.startswith(rem_label):
                    act_len = len(rem_label)
                    act_key = rem_label[0]
                    return True, act_node, act_key, act_len, ind_remainder
    return False, act_node, act_key, act_len, ind_remainder


def hop(ind, act_node, act_key, act_len, remains, ind_remainder):
    """
    Вспомогательная функция для перехода по ребру дерева
    """
    out_edge = act_node.get_out_edge(act_key)
    if not out_edge or (not act_len and not act_key):
        return act_node, act_key, act_len, ind_remainder
    anode, start, end, bnode = out_edge
    if end == '#':
        end = ind
    edge_length = end - start + 1
    while act_len > edge_length:
        # Переходим к следующему узлу по ребру, если длина ребра меньше, чем act_len
        act_node = bnode
        ind_remainder += edge_length
        act_key = remains[ind_remainder]
        act_len -= edge_length
        out_edge = act_node.get_out_edge(act_key)
        if out_edge:
            anode, start, end, bnode = out_edge
            if end == '#':
                end = ind
            edge_length = end - start + 1
        else:
            act_key = ''
            act_len = 0
            break
    if act_len == edge_length:
        act_node = bnode
        act_key = ''
        act_len = 0
        ind_remainder += edge_length
    return act_node, act_key, act_len, ind_remainder


if __name__ == "__main__":
    st = input("Введите строку: ")
    tree, pst = build(st, regularize=True)
    Node.draw(tree, pst, ed='#')
