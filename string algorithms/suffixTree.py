import string

class Arc:
  def __init__(self, i_begin, i_end, dest_vertex, i_dest_vertex):
    self.i_begin = i_begin
    self.i_end = i_end
    self.dest_vertex = dest_vertex
    self.i_dest_vertex = i_dest_vertex
    

class Node:
  def __init__(self):
    self.arcs = [None for _ in range(len(string.ascii_letters + string.digits))]


def find_suffix_tree_arc(s: str, substr: str, m: int, tree: Node) -> tuple[Arc, int, int]:
  arc = None
  idx_substr = idx_arc = 0
  curr_node = tree
  b_stopped = False

  while not b_stopped and curr_node is not None:
    next_arc = curr_node.arcs[substr[idx_substr]]
    if next_arc is not None:
      arc = next_arc
      idx_arc = arc.i_begin

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
        resullt += suffix_tree_leaves_traversal(arc)
  return result
  