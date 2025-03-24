class Node:
    ID = 0

    def __init__(self, ibeg, iend, parent=None):
        self.id = Node.ID
        Node.ID += 1
        self.ibeg: int = ibeg
        self.iend: int = iend
        self.sref: Node | None = None
        self.parent: Node | None = parent
        self.children: dict[str, Node | None] = dict()

    def __str__(self):
        return f"{self.id}) {(self.ibeg, self.iend)} {self.children.keys()} [{self.sref.id if self.sref is not None else None}]"


class Position:
    def __init__(self, tree, node, i):
        self.tree: SuffixTree = tree # чтобы иметь доступ к строке
        self.node: Node = node
        self.i: int = i

    def __str__(self):
        return f"{self.node} -- {self.i}"

    def has_move(self, ch):
        #                             на вершине                           на ребре
        return (self.i == self.node.iend and ch in self.node.children) or (self.i != self.node.iend and ch == self.tree.s[self.i + 1])

    def move(self, ch):
        # на вершине
        if self.i + 1 >= self.node.iend:
            self.node = self.node.children[ch] # спускаемся в ребёнка
            self.i = self.node.ibeg
        # на ребре
        else:
            self.i += 1 # продвигаемся по ребру

    def create_node(self, si: int):
        # si - индекс текущего символа в строке
        x = self.tree.s[si]
        n = len(self.tree.s)
        # на вершине
        if self.i == self.node.iend or self.i + 1 == self.node.iend:
            # создаём лист
            self.node.children[x] = Node(si, n, self.node)
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
            cur_node: Node = u
            while cur_node.parent is not None and cur_node.parent.sref is None:
                cur_node = u.parent
            else:
                # корень
                if cur_node.parent is None:
                    u.sref = cur_node
                # вершина с суффиксной ссылкой
                elif cur_node.parent.sref is not None:
                    cur_node = cur_node.parent.sref
                    u.sref = cur_node.children[uch]

            # усекаем старую вершину до суффикса
            self.node.parent = u
            self.node.ibeg = self.i + 1 # возможно, нужна + 1
            self.node = u

            # подвешиваем к новой вершине новый лист
            self.node.children[x] = Node(si, n, self.node)


    def go_by_sref(self):
        # old version
        self.node = self.node.sref if self.node.sref is not None else self.node
        self.i = self.node.iend


class SuffixTree:
    def __init__(self, s: str):
        self.s: str = s + '$'
        self.root: Node = Node(0, 0) # пустая подстрока в корне
        chars = set(s)
        # виртуальная вершина, чтобы работала фиктивная суффиксная ссылка и корня
        self.virtual_root: Node = Node(-1, -1)
        self.virtual_root.children = {ch: self.root for ch in chars}
        self.root.sref = self.virtual_root
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

DEBUG_I = 0
def debug_print(i: int, ch: str, pos: Position, tree: SuffixTree, desc: str = None):
    global DEBUG_I
    if True:
        print("=" * 20)
        print(f"iteration {DEBUG_I}")
        if desc is not None:
            print(desc)
        DEBUG_I += 1
        print(f"Character s[{i}] = '{ch}'")
        print(f"Position: {pos.node} --- {pos.i}")
        print(f"Tree: {tree}")
        print("=" *  20)

# st: SuffixTree = SuffixTree('abbaaba')
# print(st)
# st: SuffixTree = SuffixTree('abracadabra')
# print(st)
st: SuffixTree = SuffixTree('abcabxabcd')
print(st)
