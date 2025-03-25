from graphviz import Digraph
import sys
import time
import matplotlib.pyplot as plt
import requests
import random
import numpy as np
from scipy import stats


class Node:
    ID = 0

    def __init__(self, ibeg, iend, parent=None):
        self.id = Node.ID
        Node.ID += 1
        self.ibeg: int = ibeg
        self.iend: int = iend
        self.sref: Node | None = None
        self.isref: int | None = None
        self.parent: Node | None = parent
        self.children: dict[str, Node | None] = dict()

    def __str__(self):
        return f"{self.id}) {(self.ibeg, self.iend)} {self.children.keys()} [{(self.sref.id, self.isref) if self.sref is not None else None}]"


class Position:
    def __init__(self, tree, node, i):
        self.tree: SuffixTree = tree  # чтобы иметь доступ к строке
        self.node: Node = node
        self.i: int = i

    def __str__(self):
        return f"{self.node} -- {self.i}"

    def has_move(self, ch):
        #                             на вершине                           на ребре
        return self.i is None or (self.i + 1 >= self.node.iend and ch in self.node.children) or (
                    self.i + 1 < self.node.iend and ch == self.tree.s[self.i + 1])

    def move(self, ch):
        if self.i is None:
            return
        # на вершине
        if self.i + 1 >= self.node.iend:
            self.node = self.node.children[ch]  # спускаемся в ребёнка
            self.i = self.node.ibeg
        # на ребре
        else:
            self.i += 1  # продвигаемся по ребру

    def create_node(self, si: int):
        # si - индекс текущего символа в строке
        x = self.tree.s[si]
        n = len(self.tree.s)
        # на вершине
        if self.i == self.node.iend or self.i + 1 == self.node.iend:
            # создаём лист
            self.node.children[x] = Node(si, n, self.node)
            self.tree.iterations += 1
        # на ребре
        else:
            # РеЗнЯ РеБрА! ┻━┻ ︵ ＼( °□° )／ ︵ ┻━┻

            # выделяем вершину с общим префиксом
            u: Node = Node(self.node.ibeg, self.i + 1, self.node.parent)
            wch = self.tree.s[self.i + 1]
            # подвешиваем к новой вершине старую
            u.children[wch] = self.node

            uch = self.tree.s[u.ibeg]
            self.node.parent.children[uch] = u

            # создаём суффиксную ссылку
            cur_node: Node | None = u
            while cur_node.parent.sref is None:
                cur_node = u.parent
            else:
                cur_node = cur_node.parent.sref
                j = cur_node.ibeg
                # спускаемся
                for k in range(u.ibeg, u.iend):
                    self.tree.iterations += 1
                    uchk = self.tree.s[k]
                    if j + 1 < cur_node.iend:
                        if self.tree.s[j + 1] == uchk:
                            j += 1
                    else:
                        if uchk in cur_node.children.keys():
                            cur_node = cur_node.children[uchk]
                            j = cur_node.ibeg
                u.sref = cur_node
                u.isref = j

            # усекаем старую вершину до суффикса
            self.node.parent = u
            self.node.ibeg = self.i + 1  # возможно, нужна + 1
            self.node = u

            # подвешиваем к новой вершине новый лист
            self.node.children[x] = Node(si, n, self.node)

    def go_by_sref(self):
        # old version
        # self.node = self.node.sref if self.node.sref is not None else self.node
        # self.i = self.node.iend
        self.i = self.node.isref
        self.node = self.node.sref if self.node.sref is not None else self.node


class SuffixTree:
    def __init__(self, s: str):
        self.s: str = s + '$'
        self.iterations = 0
        self.root: Node = Node(0, 0)  # пустая подстрока в корне
        chars = set(s)
        # виртуальная вершина, чтобы работала фиктивная суффиксная ссылка и корня
        self.virtual_root: Node = Node(-1, -1)
        self.virtual_root.children = {ch: self.root for ch in chars}
        self.root.sref = self.virtual_root
        self.root.isref = self.virtual_root.ibeg
        curr: Position = Position(self, self.root, 0)  # пустая позиция в пустой подстроке корня
        debug_print(-1, '-', curr, self, "START")
        for i in range(len(self.s)):
            ch = self.s[i]
            while not curr.has_move(ch):
                curr.create_node(i)
                curr.go_by_sref()
                debug_print(i, ch, curr, self)
            curr.move(ch)
            debug_print(i, ch, curr, self, "MOVE")

    def __str__(self):
        return self.__str_node(self.root, 0)

    def __str_node(self, node, level):
        result = '\t' * level + f'{node} == "{self.s[node.ibeg: node.iend]}"\n'
        for n in node.children.values():
            result += self.__str_node(n, level + 1)
        return result

    def draw(self, filename: str = "suffix_tree") -> None:
        """Визуализирует суффиксное дерево с помощью graphviz"""
        dot = Digraph(comment='Suffix Tree', format='png')
        dot.attr(rankdir='TB')

        # Добавляем виртуальный корень отдельно
        dot.node(str(self.virtual_root.id),
                 label=f"Virtual Root\n{self.virtual_root.id}",
                 style='dashed',
                 color='gray',
                 fontcolor='gray')

        # Рекурсивная функция для обхода дерева
        def add_nodes(node: Node):
            # Создаем метку для узла
            label_lines = [
                f"ID: {node.id}",
                f"range: [{node.ibeg},{node.iend})",
                f"sref: {node.sref.id if node.sref else 'None'}",
                f"substr: '{self.s[node.ibeg:node.iend]}'"
            ]

            # Специальное оформление для корня
            if node == self.root:
                dot.node(str(node.id), '\n'.join(label_lines), shape='doublecircle')
            else:
                dot.node(str(node.id), '\n'.join(label_lines), shape='rectangle')

            # Добавляем суффиксные ссылки
            if node.sref:
                dot.edge(str(node.id),
                         str(node.sref.id),
                         style='dashed',
                         color='blue',
                         label=f" sref",
                         fontcolor='blue')

            # Рекурсивно добавляем детей
            for child in node.children.values():
                if child is not None:
                    edge_label = self.s[child.ibeg:child.iend]
                    dot.edge(str(node.id), str(child.id), label=edge_label)
                    add_nodes(child)

        # Начинаем обход с основного корня
        add_nodes(self.root)

        # Сохраняем и показываем граф
        dot.render(filename, view=True, cleanup=True)

    @staticmethod
    def _get_random_text(length: int) -> str:
        """Получает случайный текст из API или генерирует"""
        try:
            response = requests.get(
                f'https://baconipsum.com/api/?type=meat-and-filler&paras=10&format=text'
            )
            text = ' '.join(response.json()).replace('\n', ' ')[:length]
        except:
            text = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz ', k=length))
        return text

    @classmethod
    def build_performance_graph(cls, step: int = 500,
                                alpha: float = 0.05,
                                confidence_interval: float = 10):
        """Строит график с доверительными интервалами на основе статистических параметров"""
        # Рассчитываем Z-оценку
        z_score = stats.norm.ppf(1 - alpha / 2)

        lengths = list(range(100, 1001, step))
        means = []
        conf_intervals = []
        sample_sizes = []

        for length in lengths:
            # Пилотное исследование (10 замеров)
            pilot_results = []
            for _ in range(10):
                text = cls._get_random_text(length)
                tree = cls(text)
                pilot_results.append(tree.iterations)

            # Оценка стандартного отклонения
            std_estimate = np.std(pilot_results, ddof=1)

            # Расчет необходимого количества образцов
            n = int((z_score * std_estimate / confidence_interval) ** 2)
            n = max(n, 3)  # минимальное количество образцов

            # Основные замеры
            iterations = []
            for _ in range(n):
                text = cls._get_random_text(length)
                tree = cls(text)
                iterations.append(tree.iterations)

            # Сохраняем результаты
            means.append(np.mean(iterations))
            conf_intervals.append(stats.sem(iterations) * z_score)
            sample_sizes.append(n)

        # Построение графика
        plt.figure(figsize=(14, 7))
        plt.errorbar(lengths, means, yerr=conf_intervals, fmt='-o', capsize=5)
        plt.title(f"Производительность построения суффиксного дерева\n"
                  f"Доверительная вероятность: {100 * (1 - alpha)}%, "
                  f"Интервал: ±{confidence_interval} итераций")
        plt.xlabel("Длина текста")
        plt.ylabel("Среднее количество итераций")
        plt.grid(True)

        # Дополнительный график количества использованных образцов
        plt.figure(figsize=(14, 7))
        plt.plot(lengths, sample_sizes, 'r--o')
        plt.title("Использованное количество образцов для каждой длины текста")
        plt.xlabel("Длина текста")
        plt.ylabel("Количество замеров")
        plt.grid(True)

        plt.show()


DEBUG_I = 0


def debug_print(i: int, ch: str, pos: Position, tree: SuffixTree, desc: str = None):
    global DEBUG_I
    if False:
        print("=" * 20)
        print(f"iteration {DEBUG_I}")
        if desc is not None:
            print(desc)
        DEBUG_I += 1
        print(f"Character s[{i}] = '{ch}'")
        print(f"Position: {pos.node} --- {pos.i}")
        print(f"Tree: {tree}")
        print("=" * 20)


if __name__ == "__main__":
    # word = 'abbaaba'
    # word = 'abracadabra'
    # word = 'abcabxabcd'
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <word>")
        sys.exit(1)

    word = sys.argv[1]
    st: SuffixTree = SuffixTree(word)
    print(st)
    st.draw(word)
    SuffixTree.build_performance_graph(step=100, confidence_interval=3)
