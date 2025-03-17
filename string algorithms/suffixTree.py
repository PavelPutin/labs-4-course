import string

class Arc:
  def __init__(self, i_begin, i_end, dest_vertex, i_dest_vertex, src_vertex):
    self.i_begin: int = i_begin
    self.i_end: int = i_end
    self.dest_vertex: Node | None = dest_vertex
    self.i_dest_vertex: int = i_dest_vertex
    self.src_vertex: Node | None = src_vertex
    

class Node:
  def __init__(self):
    self.arcs = {ch: None for ch in str(string.ascii_letters + string.digits)}
    self.suffix_ref: Node | None = None
    self.arc_in: Arc | None  = None

def find_suffix_tree_arc(s: str, substr: str, m: int, m_same: int, tree: Node) -> tuple[Arc, int, int]:
  arc = None
  idx_substr = idx_arc = 0
  curr_node = tree
  b_stopped = False

  while not b_stopped and curr_node is not None:
    next_arc = curr_node.arcs[substr[idx_substr]]
    if next_arc is not None:
      arc = next_arc
      idx_arc = arc.i_begin

      n_same_rest = m_same - idx_substr
      if n_same_rest > 0:
        n_arc_len = arc.i_end - arc.i_begin + 1
        if n_same_rest <= n_arc_len:
          idx_substr = m_same - 1
          idx_arc += n_same_rest - 1
        else:
          idx_substr += n_arc_len
          idx_arc = arc.i_end + 1
          curr_node = arc.dest_vertex
          continue

      while idx_substr + 1 < m and idx_arc + 1 < arc.i_end + 1 and substr[idx_substr + 1] == s[idx_arc + 1]:
        idx_substr += 1
        idx_arc += 1
      
      if idx_arc <= arc.i_end:
        b_stopped = True
      else:
        curr_node = arc.dest_vertex
    else:
      b_stopped = True
  if idx_substr == m:
    idx_arc += 1
  return arc, idx_substr, idx_arc


def suffix_tree_leaves_traversal(start_arc: Arc) -> list[int]:
  result = []
  n_alpha = len(string.ascii_letters + string.digits)
  if start_arc.i_dest_vertex >= 0:
    result.append(start_arc.i_dest_vertex)
  else:
    start_node = start_arc.dest_vertex
    for k in range(n_alpha):
      arc = start_node.arcs[k]
      if arc is not None:
        result += suffix_tree_leaves_traversal(arc)
  return result


def top_jump_bottom(s: str, substr: str, m: int,
    arc: Arc, i_arc_end: int, idx_substr: int, idx_arc: int) -> tuple[Arc | None, int| None, int | None]:
  if arc is None:
    return None, None, None
  arc_next: Arc | None = None
  src_vert: Node | None = None

  is_inner_vert = arc.dest_vertex and idx_arc > i_arc_end
  src_vert = arc.dest_vertex if is_inner_vert else arc.src_vertex

  ref_vert = src_vert.suffix_ref
  if ref_vert is None:
    ref_vert = src_vert

  n_chars_up = 1 if is_inner_vert else idx_arc - arc.i_begin + 1

  if src_vert.suffix_ref is None:
    n_chars_up -= 1

  n_vert_chr = m - n_chars_up - 1
  try:
    arc_next, idx_substr, idx_arc = find_suffix_tree_arc(s, substr[n_vert_chr], n_chars_up, n_chars_up - 1, ref_vert)
  except IndexError:
    print(substr, n_vert_chr)
    raise IndexError()

  if arc_next is not None:
    arc_next = ref_vert.arc_in
    if arc_next is not None:
      idx_arc = arc_next.i_end + 1
  idx_substr += n_vert_chr
  return arc_next, idx_substr, idx_arc

def suffix_tree_build_online_1(s: str):
  n = len(s)
  tree = Node()
  arc_0 = Arc(0, 0, None, 0, tree)
  tree.arcs[s[0]] = arc_0

  js = 0
  for i in range(n):
    arc_prev = None
    i_end_prev = -1
    ref_from: Node | None = None
    idx_substr = idx_arc = 0

    for j in range(js, i + 1):
      m = i - j + 1
      if j == js:
        arc_prev = arc_0
        i_end_prev = i - 1
        idx_arc = i
        idx_substr = m - 1

      uv_arc, idx_substr, idx_arc = top_jump_bottom(s, s[j], m, arc_prev, i, idx_substr, idx_arc)
      if idx_substr == m:
        js = j
        break

      arc_prev: Arc = uv_arc
      if arc_prev is None:
        i_end_prev = -1
      elif arc_prev.dest_vertex is None:
        i_end_prev = i
      else:
        i_end_prev = arc_prev.i_end


      w_node: Node | None = None
      if uv_arc is None:
        w_node = tree
      else:
        w_node = uv_arc.dest_vertex

      if uv_arc and idx_arc <= i_end_prev:
        w_node = Node()
        w_node.arc_in = uv_arc
        wv_arc = Arc(idx_arc, uv_arc.i_end, uv_arc.dest_vertex, uv_arc.i_dest_vertex, w_node)
        w_node.arcs[s[idx_arc]] = wv_arc

        if uv_arc.dest_vertex:
          uv_arc.dest_vertex.arc_in = wv_arc

        uv_arc.dest_vertex = w_node
        uv_arc.i_end = idx_arc - 1
        uv_arc.i_dest_vertex = -1

        if ref_from is not None:
          ref_from.suffix_ref = w_node
        ref_from = w_node
      else:
        if ref_from is not None:
          ref_from.suffix_ref = w_node
          ref_from = None
      arc_0 = Arc(i, i, None, j + 1, w_node)
  return tree


suffix_tree_build_online_1('abracadabra')