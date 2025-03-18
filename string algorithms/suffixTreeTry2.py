import string


ABC: str = string.ascii_letters + string.digits


class Node:
    def __init__(self):
        self.arcs: dict[str: Arc | None] = {ch: None for ch in ABC}
        self.s_ref = None
        self.arc_in = None


class Arc:
    def __init__(self):
        self.i_beg: int | None = None
        self.i_end: int | None = None
        self.dest_vert: Node | None = None
        self.i_dest_vert: int | None = None
        self.src_vert: Node | None = None


def st_vert_init_ex(arc_in: Arc | None) -> Node:
    node = Node()
    node.arc_in = arc_in
    return node


def st_arc_init_ex(s_node: Node, ch_arc: str,
                   i_beg: int, i_end: int,
                   dest_vert: Node | None, i_dest_vert: int | None) -> Arc:
    arc = Arc()
    arc.i_beg = i_beg
    arc.i_end = i_end
    s_node.arcs[ch_arc] = arc
    arc.dest_vert = dest_vert
    arc.i_dest_vert = i_dest_vert
    arc.src_vert = s_node


def find_st_arc(s: str, substr: str,
                m: int, m_same: int,
                tree: Node,
                idx_substr: int, idx_arc: int):
    arc: Arc | None = None
    idx_substr = idx_arc = 0
    curr_node: Node | None = tree
    stopped: bool = False

    while not stopped and curr_node:
        next_arc: Arc | None = curr_node.arcs[substr[idx_substr]]
        if next_arc is not None:
            arc = next_arc
            idx_arc = arc.i_beg

            n_same_rest : int= m_same - idx_substr
            if n_same_rest > 0:
                n_arc_len: int = arc.i_end - arc.i_beg + 1
                if n_same_rest <= n_arc_len:
                    idx_substr = m_same - 1
                    idx_arc += n_same_rest - 1
                else:
                    idx_substr += n_arc_len
                    idx_arc = arc.i_end + 1
                    curr_node = arc.dest_vert
                    continue
            while idx_substr + 1 < m and idx_arc + 1 < arc.i_end + 1 and substr[idx_substr] == s[idx_arc]:
                idx_substr += 1
                idx_arc += 1

            if idx_arc <= arc.i_end:
                stopped = True
            else:
                curr_node = arc.dest_vert
        else:
            stopped = True
    if idx_substr == m:
        idx_arc += 1
    return arc, idx_substr, idx_substr


def top_jump_bottom(s: str, substr: str, m: int,
                    arc: Arc, i_arc_end: int,
                    idx_substr: int, idx_arc: int):
    if arc is None:
        return None

    arc_next: Arc | None = None
    src_vert: Node | None = None

    is_inner_vert = arc.dest_vert and idx_arc > i_arc_end
    if is_inner_vert:
        src_vert = arc.dest_vert
    else:
        src_vert = arc.src_vert

    ref_vert = src_vert.s_ref
    if ref_vert is None:
        ref_vert = src_vert

    n_chars_up: int = 1 if is_inner_vert else idx_arc - arc.i_beg + 1
    if src_vert.s_ref is None:
        n_chars_up -= 1

    n_vert_chr: int = m - n_chars_up
    arc_next, idx_substr, idx_arc = find_st_arc(s, substr[n_vert_chr],
                                                n_chars_up, n_chars_up - 1,
                                                ref_vert,
                                                idx_substr, idx_arc)
    if arc_next is None:
        arc_next = ref_vert.arc_in
        if arc_next is not None:
            idx_arc = arc_next.i_end + 1

    idx_substr += n_vert_chr
    return arc_next, idx_substr, idx_arc


def st_build_online_1(s: str):
    n: int = len(s)
    tree = st_vert_init_ex(None)
    arc_0 = st_arc_init_ex(tree, s[0], 0, 0, None, 0)
    js = 0
    for i in range(1, n):
        arc_prev