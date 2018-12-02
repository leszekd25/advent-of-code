import itertools
import operator
from math import inf

data_text = []
data = []
linecount = 0

convert = {"f": float, "s": str, "i": int}

def read_to_string():
    f = open("adventofcode_data.txt", "r")
    x = f.read()
    f.close()
    return x

# loads data and fills tables
# args: format string, default element type, element separator in a single line
def set_format(form, default = "s", elem_sep = " "):
    global data_text, data, data_format, default_format, linecount
        
    f = open("adventofcode_data.txt", "r")
    data_text = f.read().split("\n")
    f.close()
    
    data_format = form
    default_format = default
    linecount = len(data_text)
    
    for i in range(linecount):
        t = data_text[i].split(elem_sep)
        for j in range(len(t)):
            t[j] = convert[data_format[j]](t[j]) if(len(data_format) > j) else convert[default_format[0]](t[j])
        data.append(t)

# returns an element from given position
def elem(row, column=0):
    return data[row][column]

# returns a table with all elements from a given row
def column(i):
    t = []
    for j in range(linecount):
        t.append(elem(j, i))
    return t

# returns a table with all elements from a given row
def row(i):
    return data[i]

# returns a string without x first and y last characters
def cut(s, x, y):
    return s[x:-y]

# utility
# returns a filled list
def fill_list(n, v):
    l = []
    for i in range(n):
        l.append(v)
    return l

# returns a filled dict
def fill_dict(n, v):
    l = {}
    for i in range(n):
        l[i] = v
    return l

def find_key(d, val):
    for k, v in d.items():
        if(v == val):
            return k
    return None

def find_unique(l, deff = 0):
    if(len(l) < 2):
        return (None, None)
    if(len(l) == 2):
        return (deff, deff)
    x = [i for i in l]
    x.sort()
    c0 = x.count(x[0])
    c1 = x.count(x[-1])
    if(c0 == c1):
        return (None, None)
    return (x[0], x[-1]) if c0 == 1 else (x[-1], x[0])
    

# geometry stuff
# add two iterables
def v_add(v1, v2):
    return tuple(map(operator.add, v1, v2))

# subtract two iterables
def v_sub(v1, v2):
    return tuple(map(operator.sub, v1, v2))

# dot product
def v_dot(v1, v2):
    return sum(map(operator.mul, v1, v2))

# negation
def v_neg(v):
    return tuple((-x for x in v))

# multiply by scalar
def v_mul(v, r):
    return tuple((x*r for x in v))

# divide by scalar
def v_div(v, r):
    return tuple((x/r for x in v))

# snap to grid
def v_round(v):
    return tuple((int(x) for x in v))

# euclidean length
def v_lgt(v):
    return (sum((x*x for x in v)))**0.5

# square of euclidean length
def v_lgt2(v):
    return sum((x*x for x in v))

# manhattan path length from center to v (square)
def v_mpathlgt(v):
    return sum((abs(x) for x in v))

# manhattan distance from center to v (hex, square) and path length from center to v (hex)
def v_mlgt(v):
    return max((abs(x) for x in v))

# euclidean distance from v1 to v2
def v_dist(v1, v2):
    return (sum(map(lambda x, y: (x-y)*(x-y), v1, v2)))**0.5

# square of euclidean distance from v1 to v2
def v_dist2(v1, v2):
    return sum(map(lambda x, y: (x-y)*(x-y), v1, v2))

#manhattan path length from v1 to v2 (square)
def v_mpathdist(v1, v2):
    return sum(map(lambda x, y: abs(x-y), v1, v2))

# manhattan distance from v1 to v2 (hex, square) and path length from v1 to v2 (hex)
def v_mdist(v1, v2):
    return max(map(lambda x, y: abs(x-y), v1, v2))

# topology and graphs

class Topology:
    # key: node id, value: [jump count, distance, came from]
    def __init__(self):
        self.group = {}

    def __init__(self, g):
        self.group = g

    def path_to(self, i):
        if not(i in self.group):
            return None

        path = [i]
        n = i;
        while True:
            n = self.group[n][2]
            if(n == -1):
                return path
            path.append(n)

    def distance_to(self, i):
        if not(i in self.group):
            return inf

        return self.group[i][2]

    def jumps_to(self, i):
        if not(i in self.group):
            return inf

        return self.group[i][1]

    def get_nodes(self):
        return [k for k in self.group]

    def get_max_jump(self):
        return max([self.group[k][0] for k in self.group])

    def get_max_dist(self):
        return max([self.group[k][1] for k in self.group])

    def get_furthest_nodes_jump(self):
        max_jump = self.get_max_jump()
        return [k for k in self.group if self_group[k][0] == max_jump]

    def get_furthest_nodes_dist(self):
        max_dist = self.get_max_dist()
        return [k for k in self.group if self_group[k][1] == max_dist]
        

class Graph:
    class Node:
        def __init__(self):
            self.id = -1
            self.outsiders = [] # nodes which are connected to this
            self.neighbors = [] # nodes which this is connected to
            self.weights = dict()
            self.value = None

        def add(self, n, w):
            if n not in self.neighbors:
                self.neighbors.append(n)
            if self not in n.neighbors:
                n.outsiders.append(self)
            self.weights[n.id] = w

        def remove(self, n):
            if self in n.neighbors:
                n.outsiders.remove(self)
            if n in self.neighbors:
                self.neighbors.remove(n)
    
    def __init__(self, size = 0):
        self.nodes = []
        for i in range(size):
            self.add_node()

    def add_node(self, val = None):
        self.nodes.append(self.Node())
        self.nodes[len(self.nodes)-1].id = len(self.nodes)-1
        self.nodes[len(self.nodes)-1].value = val

    def connect(self, frm, to, w = 1):
        self.nodes[frm].add(self.nodes[to], w)

    def biconnect(self, frm, to, w = 1):
        self.nodes[frm].add(self.nodes[to], w)
        self.nodes[to].add(self.nodes[frm], w)
        
    def disconnect(self, frm, to):
        self.nodes[frm].remove(self.nodes[to])

    def bidisconnect(self, frm, to):
        self.nodes[frm].remove(self.nodes[to])
        self.nodes[to].remove(self.nodes[frm])

    def topology_graph(self, frm, include_distance = True):
        import heapq
        # key: id, value: [jump count, distance, came from]
        list_distances = {}
        checked = fill_list(len(self.nodes), False)
        
        # calculate jumps
        list_distances[frm] = [0, 0, -1]
        checked[frm] = True
        queue = [frm]
        while queue:
            n = queue[-1]
            del queue[-1]
            for nb in self.nodes[n].neighbors:
                if not checked[nb.id]:
                    checked[nb.id] = True
                    list_distances[nb.id] = [list_distances[n][0]+1, inf, n]
                    queue.insert(0, nb.id)

        # calculate distances
        if(include_distance):
            queue = []
            heapq.heappush(queue, (0, frm))
            while queue:
                n = heapq.heappop(queue)[1]
                for nb in self.nodes[n].neighbors:
                    new_cost = list_distances[n][1]+self.nodes[n].weights[nb.id]
                    if new_cost < list_distances[nb.id][1]:
                        list_distances[nb.id][1] = new_cost
                        heapq.heappush(queue, (new_cost, nb.id))
                        list_distances[nb.id][2] = n
            
        return Topology(list_distances)

    def find_all_roots(self):
        return [x.id for x in self.nodes if len(x.outsiders)==0]
    
    def find_all_leaves(self):
        return [x.id for x in self.nodes if len(x.neighbors)==0]

# code for 2017

def advent1_17():
    set_format("s")
    
    s = elem(0)
    
    for x in (1, len(s)//2):
        result = 0
        for i in range(len(s)):
            if(s[i] == s[(i+x)%len(s)]):
                result += int(s[i])
        print(result)

def advent2_17():
    set_format("", "i", "\t")

    print(sum([max(t)-min(t) for t in data]))
    
    s = 0
    for t in data_table:
        for i in range(len(t)):
            for j in range(i):
                d1, d2 = t[i]/t[j], t[j]/t[i]
                s += (d1 if d1 == t[i]//t[j] else (d2 if d2 == t[j]//t[i] else 0))
    print(s)

def advent3_17():
    set_format("i")

    max_w = 0
    max_h = 0
    cur_x = 0
    cur_y = 0
    d = 0  # direction

    def move():
        nonlocal max_w, max_h, cur_x, cur_y, d
        if(d == 0):
            if(cur_x == max_w):
                max_w += 1
                max_h += 1
                d = 1
            cur_x += 1            
        elif(d == 1):
            cur_y -= 1
            if(cur_y == -max_h):
                d = 2
        elif(d == 2):
            cur_x -= 1
            if(cur_x == -max_w):
                d = 3
        elif(d == 3):
            cur_y += 1
            if(cur_y == max_h):
                d = 0

    for i in range(elem(0)-1):
        move()
    print(abs(cur_x)+abs(cur_y))

    mem = {}
    max_w = 0
    max_h = 0
    cur_x = 0
    cur_y = 0
    d = 0  # direction

    mem[(0, 0)] = 1
    while True:
        move()
        summa = 0;
        for p in [(cur_x+1, cur_y), (cur_x, cur_y-1),
                  (cur_x-1, cur_y), (cur_x, cur_y+1),
                  (cur_x+1, cur_y+1), (cur_x+1, cur_y-1),
                  (cur_x-1, cur_y-1), (cur_x-1, cur_y+1)]:
            if p in mem:
                summa += mem[p]
        if(summa > elem(0)):
            raise Exception(summa)
        mem[(cur_x, cur_y)] = summa

def advent4_17():
    set_format("", "s")

    total = 0
    words = set()
    for t in data:
        words.clear()
        for w in t:
            if w in words:
                total-=1
                break
            words.add(w)
        total+=1
    print(total)

    total = 0
    for t in data:
        
        val = 1
        for i in range(len(t)):
            for j in range(i-1):
                if(sorted(t[i]) == sorted(t[j])):
                    val = 0
                    break
            if(val == 0):
                break
        total+=val
    print(total)

def advent5_17():
    set_format("i")

    for param in (0, 1):
        jump_table = [i for i in column(0)]
        cur_pos = 0
        i = 1
        while True:
            x = cur_pos+jump_table[cur_pos]
            if(x >= len(jump_table)) or (x < 0):
                cur_jump = x
                break
            jump_table[cur_pos] += (1 if (jump_table[cur_pos] < 3 or not param) else -1)
            cur_pos = x
            i+=1
        print(i)

def advent6_17():
    set_format("", "i", "\t")

    regs = [i for i in row(0)]
    reg_count = len(regs)
    reg_set = set([])
    reg_set.add(tuple(regs))
    total = 0
    repeat_set = None

    def step():
        nonlocal regs, reg_count
        max_r = 0;
        max_i = 0
        for i in range(reg_count):
            if(max_r < regs[i]):
                max_i = i
                max_r = regs[i]
        regs[max_i] = 0
        max_i += 1
        for i in range(max_r):
            regs[(i+max_i)%reg_count]+=1

    while True:
        step()
        total += 1
        regs_tuple = tuple(regs)
        if(regs_tuple in reg_set):
            repeat_set = regs_tuple
            break
        reg_set.add(regs_tuple)

    print(total)
    max_calls = total

    regs = [i for i in row(0)]
    reg_count = len(regs)
    total = 0

    while True:
        step()
        total += 1
        regs_tuple = tuple(regs)
        if(regs_tuple == repeat_set):
            raise Exception(max_calls-total)

def advent7_17():
    set_format("", "s")

    # 1: build graph from data
    node_index = {}
    p1_graph = Graph(len(data))
    for i in range(len(data)):
        t = data[i]
        node_index[t[0]] = i
    for i in range(len(data)):
        t = data[i]
        node = p1_graph.nodes[i]
        node.value = int(cut(t[1], 1, 1))
        for j in range(len(t)-3):
            p1_graph.connect(i, node_index[t[3+j].replace(",", "")])

    # 2: find root and its name
    roots = p1_graph.find_all_roots()
    name = find_key(node_index, roots[0])
    print(name)

    # 3: generate ordered tree and accumulate weights
    top = p1_graph.topology_graph(roots[0], False)
    n_table = top.get_nodes()
    total_weight = {}

    def accumulate_weight(n):
        nonlocal total_weight
        node = p1_graph.nodes[n]
        w = node.value
        for nb in node.neighbors:
            w += accumulate_weight(nb.id)
        total_weight[n] = w
        return w

    accumulate_weight(roots[0])
    
    # 4: find out which node is unbalanced
    # use the fact that the last unbalanced node found will be
    # furthest from center, so the furthest in order hierarchy
    # that's because only one node has wrong weight
    wrong_node = -1
    wrong_weight = -1
    for n in n_table:
        node = p1_graph.nodes[n]
        weights = []
        for nb in node.neighbors:
            weights.append(total_weight[nb.id])
        weights.sort(reverse = True)
        wrong_id = -1
        if(weights):
            uniq, nuniq = find_unique(weights)
            if(uniq is None) or (uniq == 0):
                continue
            for nb in node.neighbors:
                if(total_weight[nb.id] == uniq):
                    wrong_node = nb.id
                    wrong_weight = uniq
                    break

    # 5: find the correct weight
    # use the fact that only one weight is wrong,
    # so the rest are right and are the solution
    parent_node = top.group[wrong_node][2]
    node = p1_graph.nodes[wrong_node]
    p_node = p1_graph.nodes[parent_node]
    final_weights = [total_weight[nb.id] for nb in p_node.neighbors]
    uniq, nuniq = find_unique(final_weights)
    for nb in node.neighbors:
        nuniq -= total_weight[nb.id]
    print(nuniq)
    return
    
##    p1_tree = {}
##    for t in data:
##        node_children = set([])
##        for i in range(len(t)-3):
##            node_children.add(t[3+i].replace(",", ""))
##        p1_tree[t[0]] = [t[0], node_children, int(cut(t[1], 1, 1))]
##    
##    cur_node = elem(0)
##    sentinel = True
##    while sentinel:
##        print(p1_tree[cur_node])
##        sentinel = False
##        for n in p1_tree:
##            if cur_node in p1_tree[n][1]:
##                sentinel = True
##                cur_node = p1_tree[n][0]
##                break
##    print(cur_node) # now a root
##
##    def determine_weight(node):
##        total_w = p1_tree[node][2]
##        for n in p1_tree[node][1]:
##            total_w += determine_weight(n)
##        p1_tree[node].append(total_w)
##        
##        first = -1
##        summa = -len(p1_tree[node][1])
##        for n in p1_tree[node][1]:
##            if(first == -1):
##                first = p1_tree[n][3]
##                summa = 0
##            summa += p1_tree[n][3]
##        if(first*len(p1_tree[node][1]) != summa):
##            print("TROUBLE")
##            print(node, len(p1_tree[node][1]), p1_tree[node][2], total_w)
##            print("CHILDREN:")
##            for n in p1_tree[node][1]:
##                print(n, len(p1_tree[n][1]), p1_tree[n][2], p1_tree[n][3])
##
##        return total_w
##
##    determine_weight(cur_node)

def advent8_17():
    set_format("ssisssi")

    regs = dict()

    def transform(tab_line):
        nonlocal regs
        if not(tab_line[0] in regs):
            regs[tab_line[0]] = 0
        elem1 = "regs[\""+tab_line[0]+"\"] "
        elem2 = "+= " if tab_line[1] == "inc" else "-= "
        elem3 = str(tab_line[2])+" "
        elem4 = tab_line[3]+" "
        if not(tab_line[4] in regs):
            regs[tab_line[4]] = 0
        elem5 = "regs[\""+tab_line[4]+"\"] "
        elem6 = tab_line[5]+" "
        elem7 = str(tab_line[6])+" else 0 "
        return elem1+elem2+elem3+elem4+elem5+elem6+elem7

    code = ""
    max_val = 0
    for t in data:
        code += transform(t)+"\nmax_val = max((max_val, max(regs.values())))\n"
    code += "print(max_val)\n"
    exec(code)

    print(max_val)
        
def advent9_17():
    stream = read_to_string()

    def score(group):
        s = group[2]
        for g in group[4]:
            s += score(g)
        return s

    #phase 1: remove !#
    phase1 = ""
    skip = False
    for i in range(len(stream)):
        if(stream[i] == '!'):
            skip = True if not skip else False
            continue
        elif(skip == True):
            skip = False
            continue
        else:
            phase1 = phase1+stream[i]

    #phase 2: find groups
    #group is described by start, end, score, parent, and sub-groups (if any)
    #if group starts in garbage, ignore it
    cur_group = None
    cur_score = 1
    start_garbage = 0
    stop_garbage = 0
    garbage_sum = 0
    ignore_stream = False
    for i in range(len(phase1)):
        if(ignore_stream):
            if(phase1[i] == '>'):
                ignore_stream = False
                stop_garbage = i
                garbage_sum += (stop_garbage-start_garbage)-1
            continue
        if(phase1[i] == '<'):
            ignore_stream = True
            start_garbage = i
            continue
        if(phase1[i] == '{'):
            #print("START OF GROUP AT", i)
            new_group = [i, 0, cur_score, cur_group, []]
            if cur_group is not None:
                cur_group[4].append(new_group)
            cur_group = new_group
            cur_score += 1
        if(phase1[i] == '}'):
            #print("END OF GROUP AT", cur_group[0], "END AT", i)
            cur_group[1] = i
            if(cur_group[3] is not None):
                cur_group = cur_group[3]
            cur_score -= 1

    print(score(cur_group))
    print(garbage_sum)

def advent10_17():
    set_format("", "i", ",")

    knot = [i for i in range(256)]

    def reverse_slice(arr, start, lgt):
        for i in range(lgt//2):
            v1, v2 = arr[(start+i)%len(arr)], arr[(start+lgt-i-1)%len(arr)]
            arr[(start+i)%len(arr)] = v2
            arr[(start+lgt-i-1)%len(arr)] = v1

    cur_pos = 0
    skip_size = 0
    #print(knot)
    for l in row(0):
        reverse_slice(knot, cur_pos, l)
        cur_pos += l+skip_size
        skip_size += 1
        #print(cur_pos, knot)

    print(knot[0]*knot[1])

    bdata = [ord(c) for c in read_to_string()]
    bdata = bdata + [17, 31, 73, 47, 23]
    print(bdata)

    knot = [i for i in range(256)]
    cur_pos = 0
    skip_size = 0
    for i in range(64):
        for l in bdata:
            reverse_slice(knot, cur_pos, l)
            cur_pos += l+skip_size
            skip_size += 1

    xor_nums = []
    for i in range(16):
        val = knot[i*16]
        for j in range(1, 16):
            val ^= knot[i*16+j]
        xor_nums.append(val)
    print(xor_nums)

    final_str = ""
    for v in xor_nums:
        hex_str = hex(v)[2:]
        if(len(hex_str)==1):
            hex_str = '0'+hex_str
        final_str = final_str+hex_str

    print(final_str)

def advent11_17():
    set_format("", "s", ",")

    neighbors = {"n":(0, -1), "nw":(-1, 0), "sw":(-1, 1),
                 "s":(0, 1), "se":(1, 0), "ne":(1, -1)}

    max_dist = 0
    cell = (0, 0)
    for d in row(0):
        cell = v_add(cell, neighbors[d])
        max_dist = max(max_dist, v_mlgt(cell))

    print(v_mlgt(cell))
    print(max_dist)

def advent12_17():
    set_format("is", "s")

    tree = Graph(len(data))
    for t in data:
        for i in range(2, len(t)):
            tree.biconnect(t[0], int(t[i].replace(",", "")))

    total_checked = {}
    topology = tree.topology_graph(0, True)
    for i in topology.group.keys():
        total_checked[i] = i

    # print paths
    #for k in topology.group.keys():
    #    print(topology.path_to(k))

    print(len(topology.group))

    total_groups = 1
    cur_checked = 0

    while cur_checked < len(data):
        if not(cur_checked in total_checked):
            topology = tree.topology_graph(cur_checked, True)
            for i in topology.group.keys():
                total_checked[i] = i
            total_groups += 1
        cur_checked += 1

    print(total_groups)


# code for 2018

def advent1():
    set_format("i")
    
    print(sum(column(0)))
    
    frequencies = set([])
    result = 0
    for freq in itertools.cycle(data):
        result += freq[0]
        if(result in frequencies):
            raise Exception(result)
        frequencies.add(result)

def advent2():
    
    pass

#advent7_17()
