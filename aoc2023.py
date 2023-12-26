if 1:
    standard_input, packages, output_together = 1, 1, 1
    dfs, hashing, read_from_file = 0, 0, 1
    de = 1
 
    if standard_input:
        import io, os, sys
        input = lambda: sys.stdin.readline().strip()
 
        inf = float('inf')
 
        def I():
            return input()
        
        def II():
            return int(input())
 
        def MII():
            return map(int, input().split())
 
        def LI():
            return list(input().split())
 
        def LII():
            return list(map(int, input().split()))
 
        def LFI():
            return list(map(float, input().split()))
 
        def GMI():
            return map(lambda x: int(x) - 1, input().split())
 
        def LGMI():
            return list(map(lambda x: int(x) - 1, input().split()))
 
    if packages:
        from io import BytesIO, IOBase
        import math
 
        import random
        import os
 
        import bisect
        import typing
        from collections import Counter, defaultdict, deque
        from copy import deepcopy
        from functools import cmp_to_key, lru_cache, reduce
        from heapq import heapify, heappop, heappush, heappushpop, nlargest, nsmallest
        from itertools import accumulate, combinations, permutations, count, product
        from operator import add, iand, ior, itemgetter, mul, xor
        from string import ascii_lowercase, ascii_uppercase, ascii_letters
        from typing import *
        BUFSIZE = 4096
 
    if output_together:
        class FastIO(IOBase):
            newlines = 0
 
            def __init__(self, file):
                self._fd = file.fileno()
                self.buffer = BytesIO()
                self.writable = "x" in file.mode or "r" not in file.mode
                self.write = self.buffer.write if self.writable else None
 
            def read(self):
                while True:
                    b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
                    if not b:
                        break
                    ptr = self.buffer.tell()
                    self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
                self.newlines = 0
                return self.buffer.read()
 
            def readline(self):
                while self.newlines == 0:
                    b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
                    self.newlines = b.count(b"\n") + (not b)
                    ptr = self.buffer.tell()
                    self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
                self.newlines -= 1
                return self.buffer.readline()
 
            def flush(self):
                if self.writable:
                    os.write(self._fd, self.buffer.getvalue())
                    self.buffer.truncate(0), self.buffer.seek(0)
 
        class IOWrapper(IOBase):
            def __init__(self, file):
                self.buffer = FastIO(file)
                self.flush = self.buffer.flush
                self.writable = self.buffer.writable
                self.write = lambda s: self.buffer.write(s.encode("ascii"))
                self.read = lambda: self.buffer.read().decode("ascii")
                self.readline = lambda: self.buffer.readline().decode("ascii")
 
        sys.stdout = IOWrapper(sys.stdout)
 
    if dfs:
        from types import GeneratorType
 
        def bootstrap(f, stack=[]):
            def wrappedfunc(*args, **kwargs):
                if stack:
                    return f(*args, **kwargs)
                else:
                    to = f(*args, **kwargs)
                    while True:
                        if type(to) is GeneratorType:
                            stack.append(to)
                            to = next(to)
                        else:
                            stack.pop()
                            if not stack:
                                break
                            to = stack[-1].send(to)
                    return to
            return wrappedfunc
 
    if hashing:
        RANDOM = random.getrandbits(20)
        class Wrapper(int):
            def __init__(self, x):
                int.__init__(x)
 
            def __hash__(self):
                return super(Wrapper, self).__hash__() ^ RANDOM
 
    if read_from_file:
        
        import sys
        sys.stdin = open("./atom/input.txt", 'r')
        sys.stdout = open("./atom/output.txt", 'w')
 
    if de:
        def debug(*args, **kwargs):
            print('\033[92m', end='')
            print(*args, **kwargs)
            print('\033[0m', end='')

# start-----------------------------------------------------

# day 1
def aoc_1_1():
    s = I()
    ans = 0
    while len(s):
        t1,t2 = 0,0
        n = len(s)
        for i in range(n):
            if ord('0') <= ord(s[i]) <= ord('9'):
                t1 = ord(s[i]) - ord('0')
                break
        for i in range(n-1,-1,-1):
            if ord('0') <= ord(s[i]) <= ord('9'):
                t2 = ord(s[i]) - ord('0')
                break
        ans += t1 * 10 + t2
        s = I()
    print(ans)

def aoc_1_2():
    q = ['zero','one','two','three','four','five','six','seven','eight','nine']
    s = I()
    ans = 0
    while len(s):
        t1,t2 = 0,0
        n = len(s)
        for i in range(n):
            if ord('0') <= ord(s[i]) <= ord('9'):
                t1 = ord(s[i]) - ord('0')
                break
            for k,c in enumerate(q):
                if s.startswith(c,i):
                    t1 = k
                    break
            else:
                continue
            break
        for i in range(n-1,-1,-1):
            if ord('0') <= ord(s[i]) <= ord('9'):
                t2 = ord(s[i]) - ord('0')
                break
            for k,c in enumerate(q):
                if s.startswith(c,i):
                    t2 = k
                    break
            else:
                continue
            break

        ans += t1 * 10 + t2
        s = I()
    print(ans)

# day 2
def aoc_2_1():
    s = I()
    ans = 0
    ids = 1
    colors = ['red','green','blue']
    while len(s):
        f = s.index(':')
        s = s[f+1:]
        v = s.split(';')
        res = [0] * 3
        for i in v:
            d = i.split(',')
            for j in d:
                for p,t in enumerate(colors):
                    if t in j:
                        res[p] = max(res[p], int(j[:len(j)-len(t)]))
        if res[0] <= 12 and res[1] <= 13 and res[2] <= 14:
            ans += ids
        s = I()
        ids += 1
    print(ans)

def aoc_2_2():
    s = I()
    ans = 0
    ids = 1
    colors = ['red','green','blue']
    while len(s):
        f = s.index(':')
        s = s[f+1:]
        v = s.split(';')
        res = [0] * 3
        for i in v:
            d = i.split(',')
            for j in d:
                for p,t in enumerate(colors):
                    if t in j:
                        res[p] = max(res[p], int(j[:len(j)-len(t)]))
        
        ans += res[0] * res[1] * res[2]
        s = I()
        ids += 1
    print(ans)

# day 3
def aoc_3_1():
    s = I()
    ans = 0
    g = []
    dx,dy = [0,0,1,-1,1,-1,1,-1], [1,-1,0,0,1,-1,-1,1]
    while len(s):
        g.append(s)
        s = I()
    m,n = len(g),len(g[0])
    for i in range(m):
        j = 0
        while j < n:
            if g[i][j].isdigit():
                tmp = ''
                flag = False
                while j < n and g[i][j].isdigit():
                    for k in range(8):
                        x,y = i+dx[k],j+dy[k]
                        if 0<=x<m and 0<=y<n and not (g[x][y].isdigit() or g[x][y] == '.'):
                            flag = True
                    tmp += g[i][j]
                    j += 1
                if flag:
                    ans += int(tmp)
            else:
                j += 1
    print(ans)

def aoc_3_2():
    s = I()
    ans = 0
    g = []
    dx,dy = [0,0,1,-1,1,-1,1,-1], [1,-1,0,0,1,-1,-1,1]
    while len(s):
        g.append(s)
        s = I()
    m,n = len(g),len(g[0])
    mark = defaultdict(list)
    for i in range(m):
        j = 0
        while j < n:
            if g[i][j].isdigit():
                tmp = ''
                z = set()
                while j < n and g[i][j].isdigit():
                    for k in range(8):
                        x,y = i+dx[k],j+dy[k]
                        if 0<=x<m and 0<=y<n and g[x][y] == '*':
                            z.add((x,y))
                    tmp += g[i][j]
                    j += 1
                for x,y in z:
                    mark[(x,y)].append(int(tmp))
            else:
                j += 1
    for i in range(m):
        for j in range(n):
            if g[i][j] == '*':
                z = mark[(i,j)]
                
                if len(z) > 1:
                    tmp = 1
                    for v in z:
                        tmp *= v
                    ans += tmp
    print(ans)

# day 4
def aoc_4_1():
    s = I()
    ans = 0
    while len(s):
        d = s.index(':')
        s = s[d+1:]
        v = s.split('|')
        z = set()
        for i in v[0].split():
            z.add(int(i))
        p = 0
        for i in v[1].split():
            j = int(i)
            if j in z:
                p += 1
        if p:
            ans += pow(2, p - 1)
        s = I()
    print(ans)
        
def aoc_4_2():
    s = I()
    g = []
    while len(s):
        g.append(s)
        s = I()
    # 25 meybe unuse
    counts = [0] * (len(g) + 25)
    counts[0] = 1
    counts[len(g)] = -1
    ans = 0
    for idx,s in enumerate(g):
        counts[idx] += counts[idx-1]
        d = s.index(':')
        s = s[d+1:]
        v = s.split('|')
        z = set()
        for i in v[0].split():
            z.add(int(i))
        p = 0
        for i in v[1].split():
            j = int(i)
            if j in z:
                p += 1
        if p:
            counts[idx+1] += counts[idx]
            counts[idx+p+1] -= counts[idx]
    for i in range(len(g), len(counts)):
        counts[i] += counts[i-1]
    ans = sum(counts)
    print(ans)
        
# day 5
def aoc_5_1():
    s = I()
    seeds =s[6:].split()
    ss = []
    s = I()
    i = 0
    while i < 7:
        d = []
        s = I()
        while s:
            s = I()
            if s:
                v = s.split()
                d.append(v)
        ss.append(d)
        i += 1
    def f(seed, idx):
        if idx == 7:
            return seed
        for a,b,c in ss[idx]:
            if int(b) <= seed <= int(c) + int(b) - 1:
                return f(int(a) + seed - int(b), idx+1)
        return f(seed, idx+1)
    ans = inf
    for seed in seeds:
        seed = int(seed)
        ans = min(ans, f(seed, 0))
    print(ans)

def aoc_5_2():
    s = I()
    seeds =s[6:].split()
    ss = []
    s = I()
    i = 0
    while i < 7:
        d = []
        s = I()
        while s:
            s = I()
            if s:
                v = s.split()
                d.append(v)
        ss.append(d)
        i += 1
    def f(seedStart,seedEnd, idx):
        if seedStart > seedEnd:
            return inf
        if idx == 7:
            return seedStart
        res = inf
        v = [[seedStart, seedEnd]]
        nv = []
        for a,b,c in ss[idx]:
            a,b,c = int(a),int(b),int(c)
            cStart = b
            cEnd = b + c - 1
            for i,(s,e) in enumerate(v):
                if e < cStart or s > cEnd:
                    continue
                elif cStart >= s and cEnd <= e:
                    nv.append([a, a+c-1])
                    v.append([s,cStart - s - 1])
                    v.append([cEnd + 1, e])
                    v[i] = [-1,-2]
                elif cStart <= s and cEnd >= e:
                    nv.append([a + s - cStart, a + e - cStart])
                    v[i] = [-1,-2]
                # be careful
                elif cStart <= e and e <= cEnd:
                    nv.append([a, e - cStart + a])
                    v[i][1] = cStart - 1
                elif s <= cEnd and cEnd <= e:
                    nv.append([a + s - cStart, cEnd - cStart  + a])
                    v[i][0] = cEnd + 1
        for s, e in v:
            res = min(res, f(s,e, idx + 1))
        for s, e in nv:
            res = min(res, f(s,e, idx + 1))

        return res
    ans = inf
    i = 0
    while i < len(seeds):
        ans = min(ans, f(int(seeds[i]), int(seeds[i]) + int(seeds[i+1]) - 1, 0))
        i += 2
    print(ans)   

# day 6
def aoc_6_1():
    s1 = I()
    s2 = I()
    v1 = s1.split(':')[1].split()
    v2 = s2.split(':')[1].split()
    ans = 1
    for time,dist in zip(v1,v2):
        time = int(time)
        dist = int(dist)
        t1 = t2 = -1
        for i in range(time):
            if (time - i) * i >= dist:
                if t1 == -1:
                    t1 = i
                t2 = i
        ans *= (t2 - t1 + 1)
    print(ans)
        
def aoc_6_2():
    # math problem
    s1 = I()
    s2 = I()
    s1 = s1.split(':')[1]
    s1 = s1.replace(' ','')
    s2 = s2.split(':')[1]
    s2 = s2.replace(' ','')
    time  = int(s1)
    dist =  int(s2)
    t1 = (time - math.sqrt(time**2 - 4 * dist) ) / 2
    t2 = (time + math.sqrt(time**2 - 4 * dist) ) // 2
    t1 = math.ceil(t1)
    print(int(t2) - t1 + 1)


# day 7
def aoc_7_1():
    z = 'AKQJT98765432'
    class Node:
        def __init__(self, card:str ,weight:int):
           self.card = card
           self.weight = weight
        def kind(self):
            c = self.card
            v = list(c)
            v.sort()
            a = []
            t = 1
            for i in range(1, len(v)):
                if v[i] == v[i-1]:
                    t += 1
                else:
                    a.append(t)
                    t = 1
            a.append(t)
            if 5 in a: return 1
            if 4 in a: return 2
            if 3 in a and 2 in a: return 3
            if 3 in a: return 4
            if 2 == a.count(2): return 5
            if 2 in a: return 6
            return 7
            
        def __lt__(self, other):
            k1,k2 = self.kind(), other.kind()
            if k1 != k2:
                return k1 > k2
            for i in range(len(self.card)):
                if self.card[i] != other.card[i]:
                    d1,d2 = z.index(self.card[i]), z.index(other.card[i])
                    return d1 > d2
            return False

    s = I()
    v = []
    while s:
        a,b = s.split()
        node = Node(a,int(b))
        v.append(node)
        s = I()
    v.sort()
    
    ans = 0
    for i,node in enumerate(v):
        ans += (i + 1) * node.weight
    print(ans)

def aoc_7_2():
    z = 'AKQT98765432J'
    class Node:
        def __init__(self, card:str ,weight:int):
           self.card = card
           self.weight = weight
        def kind(self):
            c = self.card
            v = list(c)
            v.sort()
            a = []
            t = 1
            for i in range(1, len(v)):
                if v[i] == v[i-1]:
                    t += 1
                else:
                    a.append(t)
                    t = 1
            a.append(t)
            js = v.count('J')
            for i in range(len(a)):
                if a[i] == js:
                    a[i] = 0
                    break
            a.sort()
            a[-1] += js
            if 5 in a: return 1
            if 4 in a: return 2
            if 3 in a and 2 in a: return 3
            if 3 in a: return 4
            if 2 == a.count(2): return 5
            if 2 in a: return 6
            return 7
            
        def __lt__(self, other):
            k1,k2 = self.kind(), other.kind()
            if k1 != k2:
                return k1 > k2
            for i in range(len(self.card)):
                if self.card[i] != other.card[i]:
                    d1,d2 = z.index(self.card[i]), z.index(other.card[i])
                    return d1 > d2
            return False

    s = I()
    v = []
    while s:
        a,b = s.split()
        node = Node(a,int(b))
        v.append(node)
        s = I()
    v[3].kind()
    v.sort()
    ans = 0
    for i,node in enumerate(v):
        ans += (i + 1) * node.weight
    print(ans)

# day 8
def aoc_8_1():
    op = I()
    I()
    s = I()
    ldict = {}
    rdict = {}
    while s:
        v = s.split('=')
        v[0] = v[0].strip()
        v[1] = v[1].strip().replace('(','').replace(')','')
        q = v[1].split(',')
        q[0] = q[0].strip()
        q[1] = q[1].strip()
        ldict[v[0]] = q[0]
        rdict[v[0]] = q[1]
        s = I()
    start = 'AAA'
    end = 'ZZZ'
    step = 0
    while start != end:
        if op[step % len(op)] == 'L':
            start = ldict[start]
            step += 1
        else :
            start = rdict[start]
            step += 1
    print(step)

def aoc_8_2():
    op = I()
    I()
    s = I()
    ldict = {}
    rdict = {}
    while s:
        v = s.split('=')
        v[0] = v[0].strip()
        v[1] = v[1].strip().replace('(','').replace(')','')
        q = v[1].split(',')
        q[0] = q[0].strip()
        q[1] = q[1].strip()
        ldict[v[0]] = q[0]
        rdict[v[0]] = q[1]
        s = I()
    start = []
    for k in ldict.keys():
        if k.endswith('A'):
            start.append(k)
    res = []
    for src in start:
        step = 0
        while not src.endswith('Z'):
            if op[step % len(op)] == 'L':
                src = ldict[src]
            else :
                src = rdict[src]
            step += 1
        res.append(step)
    print(math.lcm(*res))
    
# day 9
def aoc_9_1():
    a = LII()
    def f(b):
        if len(b) == 0 or all(x == 0 for x in b):
            return 0
        n = len(b)
        z = []
        for i in range(1,n):
            z.append(b[i] - b[i-1])
        return b[-1] + f(z)
    ans = 0
    while a:
        ans += f(a)
        a = LII()
    print(ans)

def aoc_9_2():
    a = LII()
    def f(b):
        if len(b) == 0 or all(x == 0 for x in b):
            return 0
        n = len(b)
        z = []
        for i in range(1,n):
            z.append(b[i] - b[i-1])
        return b[0] - f(z)
    ans = 0
    while a:
        ans += f(a)
        a = LII()
    print(ans)

# day 10
def aoc_10_1():
    g = []
    s = I()
    while s:
        g.append(list(s))
        s = I()
    m,n = len(g),len(g[0])
    x,y = -1,-1
    for i in range(m):
        for j in range(n):
            if g[i][j] == 'S':
                x,y = i,j
    a = '|-LJ7F'
    from queue import Queue
    # 上下左右
    dx = [-1,1,0,0]
    dy = [0,0,-1,1]
    def f(x, y, c):
        
        def check(x,y):
            res = []
            if g[x][y] in '|LJ':
                nx,ny = x + dx[0], y + dy[0]
                if nx < 0 or nx >= m or ny < 0 or ny >= n or g[nx][ny] not in '|7F':
                    return []
                res.append((nx,ny))
            if g[x][y] in '|7F':
                nx,ny = x + dx[1], y + dy[1]
                if nx < 0 or nx >= m or ny < 0 or ny >= n or g[nx][ny] not in '|JL':
                    return []
                res.append((nx,ny))
            if g[x][y] in '-J7':
                nx,ny = x + dx[2], y + dy[2]
                if nx < 0 or nx >= m or ny < 0 or ny >= n or g[nx][ny] not in '-LF':
                    return []
                res.append((nx,ny))
            if g[x][y] in '-LF':
                nx,ny = x + dx[3], y + dy[3]
                if nx < 0 or nx >= m or ny < 0 or ny >= n or g[nx][ny] not in '-7J':
                    return []
                res.append((nx,ny))
            return res
        g[x][y] = c
        q = Queue()
        q.put((x, y))
        s = set()
        s.add((x, y))

        step = 0
        while not q.empty():
            sz = q.qsize()
            step += 1
            for _ in range(sz):
                x, y = q.get()
                v = check(x,y)
                if len(v) == 0:
                    continue
                t = 0
                for nx,ny in v:
                    if (nx,ny) not in s:
                        s.add((nx,ny))
                        q.put((nx,ny))
                    else:
                        t += 1
                if t == 2:
                    return step
            
        return step
            
    res = 0 
    for c in a:
        res = max(f(x,y, c),res)
    print(res)
    return

def aoc_10_2():
    g = []
    s = I()
    while s:
        g.append(list(s))
        s = I()
    m,n = len(g),len(g[0])
    x,y = -1,-1
    for i in range(m):
        for j in range(n):
            if g[i][j] == 'S':
                x,y = i,j
    a = '|-LJ7F'
    from queue import Queue
    # 上下左右
    dx = [-1,1,0,0]
    dy = [0,0,-1,1]
    def f(x, y, c):
        
        def check(x,y):
            res = []
            if g[x][y] in '|LJ':
                nx,ny = x + dx[0], y + dy[0]
                if nx < 0 or nx >= m or ny < 0 or ny >= n or g[nx][ny] not in '|7F':
                    return []
                res.append((nx,ny))
            if g[x][y] in '|7F':
                nx,ny = x + dx[1], y + dy[1]
                if nx < 0 or nx >= m or ny < 0 or ny >= n or g[nx][ny] not in '|JL':
                    return []
                res.append((nx,ny))
            if g[x][y] in '-J7':
                nx,ny = x + dx[2], y + dy[2]
                if nx < 0 or nx >= m or ny < 0 or ny >= n or g[nx][ny] not in '-LF':
                    return []
                res.append((nx,ny))
            if g[x][y] in '-LF':
                nx,ny = x + dx[3], y + dy[3]
                if nx < 0 or nx >= m or ny < 0 or ny >= n or g[nx][ny] not in '-7J':
                    return []
                res.append((nx,ny))
            return res
        g[x][y] = c
        q = Queue()
        q.put((x, y))
        s = set()
        s.add((x, y))

        step = 0
        while not q.empty():
            sz = q.qsize()
            step += 1
            for _ in range(sz):
                x, y = q.get()
                v = check(x,y)
                if len(v) == 0:
                    continue
                t = 0
                for nx,ny in v:
                    if (nx,ny) not in s:
                        s.add((nx,ny))
                        q.put((nx,ny))
                    else:
                        t += 1
                if t == 2:
                    return step
            
        return step
                      
    res = 0
    ch = '.'
    for c in a:
        k = f(x,y, c)
        if k > res:
            ch = c
            res = k
    g[x][y] = ch
    def f_c(x,y):
        def check(x,y):
            res = []
            if g[x][y] in '|LJ':
                nx,ny = x + dx[0], y + dy[0]
                if nx < 0 or nx >= m or ny < 0 or ny >= n or g[nx][ny] not in '|7F':
                    return []
                res.append((nx,ny))
            if g[x][y] in '|7F':
                nx,ny = x + dx[1], y + dy[1]
                if nx < 0 or nx >= m or ny < 0 or ny >= n or g[nx][ny] not in '|JL':
                    return []
                res.append((nx,ny))
            if g[x][y] in '-J7':
                nx,ny = x + dx[2], y + dy[2]
                if nx < 0 or nx >= m or ny < 0 or ny >= n or g[nx][ny] not in '-LF':
                    return []
                res.append((nx,ny))
            if g[x][y] in '-LF':
                nx,ny = x + dx[3], y + dy[3]
                if nx < 0 or nx >= m or ny < 0 or ny >= n or g[nx][ny] not in '-7J':
                    return []
                res.append((nx,ny))
            return res
        q = Queue()
        q.put((x, y))
        s = set()
        s.add((x, y))
        while not q.empty():
            sz = q.qsize()
            for _ in range(sz):
                x, y = q.get()
                v = check(x,y)
                if len(v) == 0:
                    continue
                for nx,ny in v:
                    if (nx,ny) not in s:
                        s.add((nx,ny))
                        q.put((nx,ny))
        ans = 0
        for i in range(m):
            for j in range(n):
                if (i,j) not in s:
                    cur = 0
                    for k in range(i,-1,-1):
                        if (k,j) in s:
                            if g[k][j] in '-7J':
                                cur += 1 
                    # 经典拓扑问题, 奇数情况下, (i,j)是在环内
                    ans += cur % 2
        return ans   

    res = f_c(x,y)
    print(res)


    return


# day 11
def aoc_11_1():
    from queue import Queue
    g = []
    s = I()
    while s:
        if '#' not in s:
            g.append(s)
        g.append(s)
        s = I()
    gc = [[] for _ in range(len(g))]
    
    for j in range(len(g[0])):
        cnt = 0
        for i in range(len(g)):
            gc[i].append(g[i][j])
            if g[i][j] == '#':
                cnt += 1
        if cnt == 0:
            for i in range(len(g)):
                gc[i].append(g[i][j])
    g = gc
    ans = 0
    z = []
    for i in range(len(g)):
        for j in range(len(g[0])):
            if g[i][j] == '#':
                z.append((i,j))
    for i in range(len(z)):
        for j in range(i+1,len(z)):
            ans += abs(z[i][0] - z[j][0]) + abs(z[i][1] - z[j][1])
    print(ans)

def aoc_11_2():
    g = []
    s = I()
    while s:
        g.append(s)
        s = I()
    m,n = len(g),len(g[0])
    row = [0] * (m + 1)
    column = [0] * (n + 1)

    for i in range(m):
        if '#' not in g[i]:
            row[i] += 1
        row[i] += row[i-1]
    for j in range(n):
        if '#' not in [g[i][j] for i in range(m)]:
            column[j] += 1
        column[j] += column[j-1]
    # 扩大倍数
    label = 1000000
    ans = 0
    z = []
    for i in range(len(g)):
        for j in range(len(g[0])):
            # be careful
            if g[i][j] == '#':
                z.append((i + row[i] * label - row[i], j + column[j] * label - column[j]))
    for i in range(len(z)):
        for j in range(i+1,len(z)):
            ans += abs(z[i][0] - z[j][0]) + abs(z[i][1] - z[j][1])
    print(ans)

# day 12
def aoc_12_1():
    def f(b:str, v:list[str]):
        z = list(b)
        def dfs(idx:int):
            if idx == len(z):
                c = []
                t = 0
                for i in range(len(z)):
                    if z[i] == '#':
                        t += 1
                    elif t:
                        c.append(t)
                        t = 0
                if t:
                    c.append(t)
                if len(c) != len(v):
                    return 0
                for x,y in zip(c,v):
                    if x != int(y):
                        return 0
                return 1
            res = 0
            if z[idx] == '.' or z[idx] == '#':
                res = dfs(idx + 1)
            else:
                z[idx] = '.'
                res += dfs(idx + 1)
                z[idx] = '#'
                res += dfs(idx + 1)
                z[idx] = '?'
            return res

        return dfs(0)
    ans = 0
    s = LI()
    while s:
        ans += f(s[0],s[1].split(','))
        s = LI()
    print(ans)

def aoc_12_2():
    def f(z:str, v:list[str]):
        v = [int(x) for x in v]
        # 动态规划
        @lru_cache(None)
        def dfs(x,y,c):
            if y >= len(v):
                if x >= len(z) or '#' not in z[x:]:
                    return 1
                return 0
            if x >= len(z):
                if c == v[y] and y == len(v) - 1:
                    return 1
                return 0
            if c > v[y]: 
                return 0
            if z[x] == '.':
                if c == v[y]: 
                    return dfs(x + 1,y + 1,0)
                elif c == 0: 
                    return dfs(x + 1,y, 0)
                return 0
            elif z[x] == '#':
                return dfs(x + 1,y,c + 1)
            else:
                res = 0
                if c == v[y]: 
                    res += dfs(x + 1,y + 1,0)
                elif c == 0: 
                    res += dfs(x + 1,y, 0)
                res += dfs(x + 1,y,c + 1)
                return res

        res = dfs(0,0,0)
        dfs.cache_clear()
        return res
    ans = 0
    s = LI()
    while s:
        b = s[0]
        c = s[1]
        for _ in range(4):
            b = b + '?' + s[0]
            c = c + ',' + s[1]
        ans += f(b,c.split(','))
        s = LI()
    print(ans)

# day 13
def aoc_13_1():
    s = I()
    ans = 0
    while s:
        g = []
        while s:
            g.append(s)
            s = I()
        row,cloumn = 0, 0
        m,n = len(g),len(g[0])
        for i in range(1,m):
            if g[i] == g[i-1]:
                l,r = i-1,i
                while l >= 0 and r < m and g[l] == g[r]:
                    l -= 1
                    r += 1
                if l < 0 or r >= m:
                    row = i
                    break
        for j in range(1,n):
            if all([g[i][j] == g[i][j-1] for i in range(m)]):
                l,r = j-1,j
                while l >= 0 and r < n and all([g[i][l] == g[i][r] for i in range(m)]):
                    l -= 1
                    r += 1
                if l < 0 or r >= n:
                    cloumn = j
                    break
        ans += row * 100 + cloumn
        s = I()
    print(ans)

def aoc_13_2():
    s = I()
    ans = 0
    while s:
        g = []
        while s:
            g.append(s)
            s = I()
        row = 0
        column = 0
        m,n = len(g),len(g[0])
        # one difference
        for i in range(1,m):
            cnt = 0
            if sum([int(g[i][j] != g[i-1][j]) for j in range(n)]) == 1:
                l,r = i-2,i+1
                while l >= 0 and r < m and g[l] == g[r]:
                    l -= 1
                    r += 1
                if l < 0 or r >= m:
                    row = i
                    break
            if g[i] == g[i-1]:
                l,r = i-1,i
                while l >= 0 and r < m:
                    cnt += sum([int(g[l][j] != g[r][j]) for j in range(n)]) 
                    l -= 1
                    r += 1
                if cnt == 1:
                    row = i
                    break
        for j in range(1,n):
            cnt = 0
            if all([g[i][j] == g[i][j-1] for i in range(m)]):
                l,r = j-1,j
                while l >= 0 and r < n:
                    cnt += sum([int(g[i][l] != g[i][r]) for i in range(m)])
                    l -= 1
                    r += 1
                if cnt == 1:
                    column = j
                    break
            if sum([int(g[i][j] != g[i][j-1]) for i in range(m)]) == 1:
                l,r = j-2,j+1
                while l >= 0 and r < n and all([g[i][l] == g[i][r] for i in range(m)]):
                    l -= 1
                    r += 1
                if l < 0 or r >= n:
                    column = j
                    break
        ans += row * 100  + column
        s = I()
    print(ans)

# day 14
def aoc_14_1():
    g = []
    s = I()
    while s:
        g.append(list(s))
        s = I()
    m,n = len(g),len(g[0])
    for j in range(n):
        l = r = 0
        while r < m:
            if g[r][j] == '.':
                pass
            elif g[r][j] == '#':
                l = r + 1
            else:
                g[l][j],g[r][j] = g[r][j],g[l][j]
                l += 1
            r += 1

    ans = 0
    for i in range(m):
        ans += g[i].count('O') * (m - i)
    print(ans)

def aoc_14_2():
    # TODO 
    # 输出完之后, 接下来使用瞪眼法吧(
    # 一种思路: 找一个大的循环, 可能包含多个小循环,不过不要紧
    g = []
    s = I()
    while s:
        g.append(list(s))
        s = I()
    m,n = len(g),len(g[0])
    def north():
        for j in range(n):
            l = r = 0
            while r < m:
                if g[r][j] == '.':
                    pass
                elif g[r][j] == '#':
                    l = r + 1
                else:
                    g[l][j],g[r][j] = g[r][j],g[l][j]
                    l += 1
                r += 1
    def south():
        for j in range(n):
            l = r = m-1
            while r >= 0:
                if g[r][j] == '.':
                    pass
                elif g[r][j] == '#':
                    l = r - 1
                else:
                    g[l][j],g[r][j] = g[r][j],g[l][j]
                    l -= 1
                r -= 1
    def west():
        for i in range(m):
            l = r = 0
            while r < n:
                if g[i][r] == '.':
                    pass
                elif g[i][r] == '#':
                    l = r + 1
                else:
                    g[i][l],g[i][r] = g[i][r],g[i][l]
                    l += 1
                r += 1
    def east():
        for i in range(m):
            l = r = n-1
            while r >= 0:
                if g[i][r] == '.':
                    pass
                elif g[i][r] == '#':
                    l = r - 1
                else:
                    g[i][l],g[i][r] = g[i][r],g[i][l]
                    l -= 1
                r -= 1
    
    cyc = 1000000000
    t = []
    for i in range(1, 1000):
        north()
        west()
        south()
        east()
        ans = 0
        for j in range(m):
            ans += g[j].count('O') * (m - j)
        t.append(ans)
        print(ans)
   
# day 15
def aoc_15_1():
    s = I()
    s = s.split(',')
    ans = 0
    for v in s:
        cur = 0
        for c in v:
            cur += ord(c)
            cur *= 17
            cur %= 256
        ans += cur
    print(ans)

def aoc_15_2():

    s = I()
    s = s.split(',')
    ans = 0
    mp = [[] for _ in range(256)]
    for v in s:
        d = []
        if '=' in v:
            d = v.split('=')
        else:
            d.append(v[:-1])
        
        if len(d) == 2: # =
            cur = 0
            for c in d[0]:
                cur += ord(c)
                cur *= 17
                cur %= 256
            for i,(k,v,de) in enumerate(mp[cur]):
                if k == d[0] and de == False:
                    mp[cur][i] = (k,d[1],False)
                    break
            else:
                mp[cur].append((d[0],d[1],False))
        else: # -
            cur = 0
            for c in d[0]:
                cur += ord(c)
                cur *= 17
                cur %= 256
            for i,(k,v,de) in enumerate(mp[cur]):
                if k == d[0] and de == False:
                    mp[cur][i] = (k,v,True)
    ans = 0
    for i in range(256):
        d = 1
        for (k,v,de) in mp[i]:
            if not  de:
                ans += (i + 1) * d * (int(v))
                d += 1
    print(ans)

# day 16
def aoc_16_1():
    g = []
    s = I()
    while s:
        g.append(s)
        s = I()
    m,n = len(g),len(g[0])
    # 上下左右
    # 0 1 2 3
    z = [set() for _ in range(4)]
    dx = [-1,1,0,0]
    dy = [0,0,-1,1]
    def f(x, y, dir):
        if x < 0 or x >= m or y < 0 or y >= n:
            return []
        if g[x][y] == '.':
            return [(x + dx[dir], y + dy[dir], dir)]
        if g[x][y] == '|' and (dir == 0 or dir == 1):
            return [(x + dx[dir], y + dy[dir], dir)]
        if g[x][y] == '-' and (dir == 2 or dir == 3):
            return [(x + dx[dir], y + dy[dir], dir)]
        if g[x][y] == '|' and (dir == 2 or dir == 3):
            return [(x + dx[0], y + dy[0], 0), (x + dx[1], y + dy[1], 1)]
        if g[x][y] == '-' and (dir == 0 or dir == 1):
            return [(x + dx[2], y + dy[2], 2), (x + dx[3], y + dy[3], 3)]
        if g[x][y] == '/' and (dir == 0 or dir == 2):
            return [(x + dx[(dir - 1) % 4], y + dy[(dir - 1) % 4], (dir - 1) % 4)]
        if g[x][y] == '/' and (dir == 1 or dir == 3):
            return [(x + dx[(dir + 1) % 4], y + dy[(dir + 1) % 4], (dir + 1) % 4)]
        if g[x][y] == '\\' and dir == 0:
            return [(x + dx[2], y + dy[2], 2)]
        if g[x][y] == '\\' and dir == 1:
            return [(x + dx[3], y + dy[3], 3)]
        if g[x][y] == '\\' and dir == 2:
            return [(x + dx[0], y + dy[0], 0)]
        if g[x][y] == '\\' and dir == 3:
            return [(x + dx[1], y + dy[1], 1)]
    
    import queue
    q = queue.Queue()
    q.put((0,0,3))
    z[3].add((0,0))
    while not q.empty():
        x,y,dir = q.get()
        v = f(x,y,dir)
        for x1,y1,dir1 in v:
            if x1 < 0 or x1 >= m or y1 < 0 or y1 >= n:
                continue
            if (x1,y1) not in z[dir1]:
                z[dir1].add((x1,y1))
                q.put((x1,y1,dir1))
    ans = 0
    for i in range(m):
        for j in range(n):
            for k in range(4):
                if (i,j) in z[k]:
                    ans += 1
                    break
    print(ans)

def aoc_16_2():
    g = []
    s = I()
    while s:
        g.append(s)
        s = I()
    m,n = len(g),len(g[0])
    # 上下左右
    # 0 1 2 3
    dx = [-1,1,0,0]
    dy = [0,0,-1,1]
    def w(x,y,dir):
        # 上下左右
        z = [set() for _ in range(4)]
        def f(x, y, dir):
            if x < 0 or x >= m or y < 0 or y >= n:
                return []
            if g[x][y] == '.':
                return [(x + dx[dir], y + dy[dir], dir)]
            if g[x][y] == '|' and (dir == 0 or dir == 1):
                return [(x + dx[dir], y + dy[dir], dir)]
            if g[x][y] == '-' and (dir == 2 or dir == 3):
                return [(x + dx[dir], y + dy[dir], dir)]
            if g[x][y] == '|' and (dir == 2 or dir == 3):
                return [(x + dx[0], y + dy[0], 0), (x + dx[1], y + dy[1], 1)]
            if g[x][y] == '-' and (dir == 0 or dir == 1):
                return [(x + dx[2], y + dy[2], 2), (x + dx[3], y + dy[3], 3)]
            if g[x][y] == '/' and (dir == 0 or dir == 2):
                return [(x + dx[(dir - 1) % 4], y + dy[(dir - 1) % 4], (dir - 1) % 4)]
            if g[x][y] == '/' and (dir == 1 or dir == 3):
                return [(x + dx[(dir + 1) % 4], y + dy[(dir + 1) % 4], (dir + 1) % 4)]
            if g[x][y] == '\\' and dir == 0:
                return [(x + dx[2], y + dy[2], 2)]
            if g[x][y] == '\\' and dir == 1:
                return [(x + dx[3], y + dy[3], 3)]
            if g[x][y] == '\\' and dir == 2:
                return [(x + dx[0], y + dy[0], 0)]
            if g[x][y] == '\\' and dir == 3:
                return [(x + dx[1], y + dy[1], 1)]
        
        import queue
        q = queue.Queue()
        q.put((x,y,dir))
        z[dir].add((x,y))
        while not q.empty():
            x,y,dir = q.get()
            v = f(x,y,dir)
            for x1,y1,dir1 in v:
                if x1 < 0 or x1 >= m or y1 < 0 or y1 >= n:
                    continue
                if (x1,y1) not in z[dir1]:
                    z[dir1].add((x1,y1))
                    q.put((x1,y1,dir1))
        ans = 0
        for i in range(m):
            for j in range(n):
                for k in range(4):
                    if (i,j) in z[k]:
                        ans += 1
                        break
        return ans
    
    res = 0
    for i in range(m):
        res = max(res,w(i,0,3))
        res = max(res,w(i,n-1,2))
    for i in range(n):
        res = max(res,w(0,i,1))
        res = max(res,w(m-1,i,0))

    print(res)

# day 17
def aoc_17_1():
    sys.setrecursionlimit(1000000)
    from queue import PriorityQueue
    g = []
    s = I()
    while s:
        g.append([int(i) for i in s])
        s = I()
    m,n = len(g),len(g[0])
    INF = 1_000_000_000_000
    # 上下左右
    dx = [-1,1,0,0]
    dy = [0,0,-1,1]
    ds = {0:[0,2,3], 1:[1,2,3], 2:[0,1,2], 3:[3,0,1]}
    def back(dir):
        dp = [[[[INF for _ in range(4)] for _ in range(4)] for _ in range(n)] for _ in range(m)]
        pq = PriorityQueue()
        # cost x y dir cnt
        pq.put((0,0,0,dir,1))
        while not pq.empty():
            cost,x,y,dir,cnt = pq.get()
            if x < 0 or x >= m or y < 0 or y >= n or cnt > 3: 
                continue
            if cost >= dp[x][y][dir][cnt]:
                continue
            dp[x][y][dir][cnt] = cost
            for ndir in ds[dir]:
                nx, ny = x + dx[ndir], y + dy[ndir]
                if ndir == dir:   
                    pq.put((cost + g[x][y], nx, ny, ndir, cnt + 1))
                else:
                    pq.put((cost + g[x][y], nx, ny, ndir, 1))
        
        ans = min([dp[-1][-1][i][j] for i in range(4) for j in range(4)]) -g[0][0] + g[-1][-1]
        #print(ans)
        return ans
    res = INF
    for i in range(4):
        res = min(res,back(i))
    print(res) 

def aoc_17_2():
    sys.setrecursionlimit(1000000)
    from queue import PriorityQueue
    g = []
    s = I()
    while s:
        g.append([int(i) for i in s])
        s = I()
    m,n = len(g),len(g[0])
    INF = 1_000_000_000_000
    # 上下左右
    dx = [-1,1,0,0]
    dy = [0,0,-1,1]
    ds = {0:[0,2,3], 1:[1,2,3], 2:[0,1,2], 3:[3,0,1]}
    def back(dir):
        dp = [[[[INF for _ in range(11)] for _ in range(4)] for _ in range(n)] for _ in range(m)]
        pq = PriorityQueue()
        # cost x y dir cnt
        pq.put((0,0,0,dir,1))
        while not pq.empty():
            cost,x,y,dir,cnt = pq.get()
            if x < 0 or x >= m or y < 0 or y >= n or cnt > 10: 
                continue
            if cost >= dp[x][y][dir][cnt]:
                continue
            dp[x][y][dir][cnt] = cost
            if cnt < 4:
                pq.put((cost + g[x][y], x + dx[dir], y + dy[dir], dir, cnt + 1))
            else:
                for ndir in ds[dir]:
                    nx, ny = x + dx[ndir], y + dy[ndir]
                    if ndir == dir:   
                        pq.put((cost + g[x][y], nx, ny, ndir, cnt + 1))
                    else:
                        pq.put((cost + g[x][y], nx, ny, ndir, 1))
        
        ans = min([dp[-1][-1][i][j] for i in range(4) for j in range(11)]) -g[0][0] + g[-1][-1]
        #print(ans)
        return ans
    res = INF
    for i in range(4):
        res = min(res,back(i))
    print(res) 


# day 18
def aoc_18_1():
    s = I()
    z = []
    while s:
        s = s.split()
        z.append((s[0], int(s[1])))
        s = I()
    sx,sy,ex,ey = 0,0,0,0
    cx,cy = 0,0
    
    # 上下左右
    dx = [-1,1,0,0]
    dy = [0,0,-1,1]
    ds = {'U':0,'D':1,'L':2,'R':3}
    used = set()
    for dir, step in z:
        d = ds[dir]
        for _ in range(step):
            cx += dx[d]
            cy += dy[d]
            sx,sy = min(sx,cx),min(sy,cy)
            ex,ey = max(ex,cx),max(ey,cy)
            used.add((cx,cy))
    from queue import Queue
    q = Queue()
    s = set()
    for i in range(sx,ex+1):
        if (i,sy) not in used:
            q.put((i,sy))
            s.add((i,sy))
        if (i,ey) not in used:
            q.put((i,ey))
            s.add((i,ey))
    for j in range(sy,ey+1):
        if (sx,j) not in used:
            q.put((sx,j))
            s.add((sx,j))
        if (ex,j) not in used:
            q.put((ex,j))
            s.add((ex,j))
    while not q.empty():
        x,y = q.get()
        for d in range(4):
            nx,ny = x + dx[d],y + dy[d]
            if (nx,ny) not in used and (nx,ny) not in s and  sx <= nx <= ex and sy <= ny <= ey:
                q.put((nx,ny))
                s.add((nx,ny))
    ans = (ex-sx+1) * (ey-sy+1) - len(s)
    print(ans)
    
def aoc_18_2():
    s = I()
    z = []
    while s:
        s = s.split()[-1]
        s = s[2:-1]
        z.append((s[-1:],int(s[:-1],16)))
        s = I()

    def _solve(lines):
        x, y, a, l = 0, 0, 0, 0
        for d, n in lines:
            match d:
                case "0" | "R":
                    x += n
                    a += y * n
                case "1" | "D":
                    y += n
                case "2" | "L":
                    x -= n
                    a -= y * n
                case "3" | "U":
                    y -= n
            l += n
        return abs(a) + l // 2 + 1
    ans = _solve(z)
    
    
    print(ans)


# day 19
def aoc_19_1():
    # workflow
    s = I()
    g = defaultdict(list)
    while s:
        s = s.split('{')
        s[1] = s[1][:-1]
        v = s[1].split(',')
        for i in range(len(v)):
            g[s[0]].append(v[i])
        s = I()
    # part
    s = I()
    z = []
    while s:
        s = s[1:-1]
        s = s.split(',')
        for i in range(len(s)):
            s[i] = int(s[i][2:])
        z.append(s)
        s = I()
    ans = 0
    for x,m,a,s in z:
        
        start = 'in'
        while start != 'A' and start != 'R':
            cur = g[start]
            for i in range(len(cur) - 1):
                d = -1
                if cur[i][0] == 'x':
                    d = x
                elif cur[i][0] == 'm':
                    d = m
                elif cur[i][0] == 'a':
                    d = a
                elif cur[i][0] =='s':
                    d = s
                q  = cur[i].split(':')
                e = int(q[0][2:])
                if q[0][1] == '<':
                    if d < e:
                        start = q[1]
                        break
                else:
                    if d > e:
                        start = q[1]
                        break

            else:
                start = cur[-1]
        if start == 'A':
            ans += x + m + a + s
    print(ans)


    return

def aoc_19_2():
    # workflow
    s = I()
    g = defaultdict(list)
    while s:
        s = s.split('{')
        s[1] = s[1][:-1]
        v = s[1].split(',')
        for i in range(len(v)):
            g[s[0]].append(v[i])
        s = I()
    
    def f(x,m,a,s,state):
        if state == 'R':
            return 0
        if state == 'A':
            return (x[1] - x[0] + 1) * (m[1] - m[0] + 1) * (a[1] - a[0] + 1) * (s[1] - s[0] + 1)
        cur = g[state]
        x,m,a,s = list(x),list(m),list(a),list(s)
        res = 0
        for i in range(len(cur) - 1):
            d = []
            if cur[i][0] == 'x':d = x
            elif cur[i][0] == 'm':d = m
            elif cur[i][0] == 'a':d = a
            elif cur[i][0] =='s':d = s
            q  = cur[i].split(':')
            e = int(q[0][2:])

            if q[0][1] == '<':
                if d[0] >= e:
                    continue
                elif d[1] < e:
                    res += f(tuple(x),tuple(m),tuple(a),tuple(s),q[1])
                    d[0],d[1] = 0,-1
                else:
                    us,ue = d[0],d[1]
                    d[1] = e-1
                    res += f(tuple(x),tuple(m),tuple(a),tuple(s),q[1])
                    d[0],d[1] = e,ue
            else:
                if d[1] <= e:
                    continue
                elif d[0] > e:
                    res += f(tuple(x),tuple(m),tuple(a),tuple(s),q[1])
                    d[0],d[1] = 0,-1
                else:
                    us,ue = d[0],d[1]
                    d[0] = e+1
                    res += f(tuple(x),tuple(m),tuple(a),tuple(s),q[1])
                    d[0],d[1] = us,e
        else:
            res += f(tuple(x),tuple(m),tuple(a),tuple(s),cur[-1])
        return res
    ans = f((1,4000),(1,4000),(1,4000),(1,4000),'in')
    print(ans)

    
# day 20
def aoc_20_1():
    s = I()
    g = defaultdict(list)
    dg = defaultdict(list)
    state = defaultdict(bool)
    con = set()
    while s:
        v = s.split(' -> ')
        f,t = v[0],v[1].split(',')
        if f[0] == '%':
            state[f[1:]] = False
            for i in range(len(t)):
                t[i] = t[i].strip()
                g[f[1:]].append(t[i])
                dg[t[i]].append(f[1:])
        elif f[0] == '&':
            con.add(f[1:])
            for i in range(len(t)):
                t[i] = t[i].strip()
                g[f[1:]].append(t[i])
                dg[t[i]].append(f[1:])
        else:
            state[f] = True
            for i in range(len(t)):
                t[i] = t[i].strip()
                g[f[0:]].append(t[i])
        s = I()
    start = 'broadcaster'
    from queue import Queue
    c = defaultdict(set)
    def f():
        q = Queue()
        state['broadcaster'] = True
        q.put(('broadcaster', False))
        low,high = 0,0
        
        while not q.empty():
            cur,st = q.get()
            if cur in con: # &
                nst = True

                if len(dg[cur]) == len(c[cur]):
                    nst = False
                #del c[cur]
                for nxt in g[cur]:
                    q.put((nxt,nst))
                    
                    if nst:
                        if nxt in con:
                            c[nxt].add(cur)
                        high += 1
                    else:
                        if nxt in con:
                            if cur in c[nxt]:
                                c[nxt].remove(cur)
                        low += 1
            else:          # % or boardcaster
                if not st: # low
                    state[cur] = not state[cur]
                    for nxt in g[cur]:
                        q.put((nxt, state[cur]))
                        if state[cur]:
                            if nxt in con:
                                c[nxt].add(cur)
                            high += 1
                        else:
                            if nxt in con:
                                if cur in c[nxt]:
                                    c[nxt].remove(cur)

                            low += 1
                else:      # high
                    pass
        return (low,high)
    r1,r2 = 0,0
    for _ in range(1000):
        l1,l2 = f()
        r1,r2 = r1 + l1 + 1, r2 + l2
    print(r1*r2)





        
            

    return

def aoc_20_2():
    # 这一问需要根据数据特点来做！！！
    s = I()
    g = defaultdict(list)
    dg = defaultdict(list)
    state = defaultdict(bool)
    con = set()
    while s:
        v = s.split(' -> ')
        f,t = v[0],v[1].split(',')
        if f[0] == '%':
            state[f[1:]] = False
            for i in range(len(t)):
                t[i] = t[i].strip()
                g[f[1:]].append(t[i])
                dg[t[i]].append(f[1:])
        elif f[0] == '&':
            con.add(f[1:])
            for i in range(len(t)):
                t[i] = t[i].strip()
                g[f[1:]].append(t[i])
                dg[t[i]].append(f[1:])
        else:
            state[f] = True
            for i in range(len(t)):
                t[i] = t[i].strip()
                g[f[0:]].append(t[i])
        s = I()

    from queue import Queue
    c = defaultdict(set)
    res = 1
    cnt = 0
    rrr = dict()
    def f(i):
        nonlocal res,cnt
        q = Queue()
        state['broadcaster'] = True
        q.put(('broadcaster', False))
        low,high = 0,0
        cnt = 0
        while not q.empty():
            cur,st = q.get()
    

            if cur in con: # &
                nst = True
                if len(dg[cur]) == len(c[cur]):
                    nst = False
                #del c[cur]
                for nxt in g[cur]:
                    q.put((nxt,nst))
                    if nxt == 'mg' and cur not in rrr and nst:
                        rrr[cur] = i
                    if nst:
                        if nxt in con:
                            c[nxt].add(cur)
                        high += 1
                    else:
                        if nxt in con:
                            if cur in c[nxt]:
                                c[nxt].remove(cur)
                        low += 1
            else:          # % or boardcaster
                if not st and len(g[cur]) > 0: # low
                    state[cur] = not state[cur]
                    for nxt in g[cur]:
                        q.put((nxt, state[cur]))
                        if nxt == 'mg' and cur not in rrr and state[cur]:
                            rrr[cur] = i
                        if state[cur]:
                            if nxt in con:
                                c[nxt].add(cur)
                            high += 1
                        else:
                            if nxt in con:
                                if cur in c[nxt]:
                                    c[nxt].remove(cur)

                            low += 1
                else:      # high
                    pass
        return cnt
    
    for i in range(1,1000000):
        r =  f(i)
        if len(rrr) == 4:
            break
    print(math.prod(list(rrr.values())))

# day 21
def aoc_21_1():
    s = I()
    g = []
    while s:
        g.append(list(s))
        s = I()
    m,n = len(g),len(g[0])
    def f():
        nxt = set()
        cur = set()
        for i in range(m):
            for j in range(n):
                if g[i][j] == 'O' or g[i][j] == 'S':
                    cur.add((i,j))
        dx = [0,1,0,-1]
        dy = [1,0,-1,0]
        for (x, y) in cur:
            for i in range(4):
                nx = x + dx[i]
                ny = y + dy[i]
                if nx < 0 or nx >= m or ny < 0 or ny >= n:
                    continue
                if g[nx][ny] == '.':
                    nxt.add((nx,ny))
            g[x][y] = '.'
        for (x, y) in nxt:
            g[x][y] = 'O'
                
        return len(nxt)
    
    ans = 0
    for _ in range(64):
        ans = f()
    # for i in range(m):
    #     print(''.join(g[i]))
    print(ans)

def aoc_22_2():
    return "I can't do that"
    


# day 22
def aoc_22_1_2():
    
    import re
    def absrange(a,b): return range(a,b+1)
    class Block:
        def __init__(self,x,y,z,b): self.x,self.y,self.z,self.brick = x,y,z,b
        def is_supported(self): return self.z==1 or (blocks.get((self.x,self.y,self.z-1), self).brick != self.brick)

    class Brick:
        def __init__(self,s,e): self.blocks = [Block(x,y,z,self) for x in absrange(s[0],e[0]) for y in absrange(s[1],e[1]) for z in absrange(s[2],e[2])]
        def is_falling(self): return not any(b.is_supported() for b in self.blocks)

    def collapse(bricks):
        dropped = set()
        for br in bricks:
            while br.is_falling():
                for b in br.blocks:
                    blocks[b.x,b.y,b.z-1] = blocks.pop((b.x,b.y,b.z))
                    b.z -= 1
                dropped.add(br)
        return len(dropped)
    aocinput = []
    s = I()
    while s:
        aocinput.append(s)
        s = I()
    aocdata = sorted([[int(x) for x in re.findall('(\d+)', line)] for line in aocinput], key=lambda a:min(a[2],a[5]))
    bricks = [Brick((a,b,c),(d,e,f)) for a,b,c,d,e,f in aocdata]
    blocks = {(b.x,b.y,b.z): b for br in bricks for b in br.blocks}
    collapse(bricks)
    saveloc = {b:k for k,b in blocks.items()}

    part1 = part2 = 0
    for i,br in enumerate(bricks):
        for b in saveloc: b.x,b.y,b.z = saveloc[b]
        blocks = {saveloc[b]:b for b in saveloc if b.brick != br}
        dropped = collapse(bricks[:i]+bricks[i+1:])
        part1 += dropped==0
        part2 += dropped
    print(part1,part2)

# day 23
def aoc_23_1():
    sys.setrecursionlimit(1000000)
    g = []
    s = I()
    while s:
        g.append(list(s))
        s = I()
    m,n = len(g),len(g[0])
    start = (0, g[0].index('.'))
    end = (m-1, g[m-1].index('.'))
    # 上下左右
    dx = [-1,1,0,0]
    dy = [0,0,-1,1]
    INF = 1_000_000_000
    used = set()   
    
    def f(src,tar):
        if src[0] < 0 or src[0] >= m or src[1] < 0 or src[1] >= n or g[src[0]][src[1]] == '#' or (src[0], src[1]) in used:
            return -INF
        if src == tar:
            return 0
        used.add(src)
        res = -INF
        match g[src[0]][src[1]]:
            case '^':
                res = 1 + f((src[0]-1,src[1]),tar)
            case 'v':
                res = 1 + f((src[0]+1,src[1]),tar)
            case '<':
                res = 1 + f((src[0],src[1]-1),tar)
            case '>':
                res = 1 + f((src[0],src[1]+1),tar)
        if g[src[0]][src[1]] in '^v<>':
            used.remove(src)
            return res
        for i in range(4):
            nx = src[0] + dx[i]
            ny = src[1] + dy[i]
            res = max(res, 1 + f((nx,ny),tar))
        used.remove(src)
        return res
    ans = f(start,end)
    print(ans)

def aoc_23_2():
    # 枚举分叉点
    sys.setrecursionlimit(1000000)
    g = []
    s = I()
    while s:
        g.append(list(s))
        s = I()
    m,n = len(g),len(g[0])
    start = (0, g[0].index('.'))
    end = (m-1, g[m-1].index('.'))
    # 上下左右
    dx = [-1,1,0,0]
    dy = [0,0,-1,1]
    tmp = []
    for i in range(m):
        for j in range(n):
            if g[i][j] != '#':
                cnt = 0
                for k in range(4):
                    nx = i + dx[k]
                    ny = j + dy[k]
                    if nx < 0 or nx >= m or ny < 0 or ny >= n or g[nx][ny] == '#':
                        continue
                    cnt += 1
                if cnt > 2:
                    tmp.append((i,j))
    tmp.append(start)
    tmp.append(end)
    ans = 0
    pts = set(tmp)
    path = defaultdict(lambda: Counter())
    for x,y in tmp:
        a,b = x,y
        dq = deque([(x,y)])
        dist = Counter()
        dist[(x,y)] = 0
        while dq:
            x,y = dq.popleft()
            for i in range(4):
                nx = x + dx[i]
                ny = y + dy[i]
                if 0 <= nx < m and 0 <= ny < n and g[nx][ny]!= '#' and (nx,ny) not in dist:
                    if (nx,ny) in pts:
                        path[(a,b)][(nx,ny)] = dist[(x,y)] + 1
                    else:
                        dq.append((nx,ny))
                        dist[(nx,ny)] = dist[(x,y)] + 1
    vis = Counter()
    vis[start] =  1
    ans = 0
    def dfs(u, cur):
        nonlocal ans
        if u == end:
            ans = max(ans, cur)
            return
        for v in path[u]:
            if not vis[v]:
                vis[v] = 1
                dfs(v, cur + path[u][v])
                vis[v] = 0
    dfs(start, 0)
    print(ans)

# day 24
def aoc_24_1():
    import sympy as sp
    g = []
    s = I()
    while s:
        v = s.replace('@',',').replace(' ','').split(',')
        v = [int(x) for x in v]
        g.append(v)
        s = I()
    ans = 0
    #start,end = 200000000000000,400000000000000
    start,end = 200000000000000,400000000000000

    for i,(x,y,z,vx,vy,vz) in enumerate(g):
        t1 = sp.Symbol('t1')
        x1 = x + vx * t1
        y1 = y + vy * t1
        for j, (x,y,z,vx,vy,vz)in enumerate(g[i+1:]):
            t2 = sp.Symbol('t2')
            x2 = x + vx * t2
            y2 = y + vy * t2
            sol = sp.solve((x1-x2,y1-y2),(t1,t2))
            if len(sol):
                x = x + vx * sol[t2]
                y = y + vy * sol[t2]
                if sol[t2] >= 0 and sol[t1] >= 0 and start <= x <= end and start <= y <= end:
                    ans += 1
    print(ans)
            
    # from matplotlib import pyplot as plt
    # import numpy as np
    # # 绘制曲线
    # def curve(x,y):
    #     plt.plot(x,y)
    # for x,y,z,vx,vy,vz in g:
    #     x = x + vx * np.linspace(0,20,100)
    #     y = y + vy * np.linspace(0,20,100)
    #     curve(x,y)
    # plt.show()

def aoc_24_2():
    import sympy as sp
    eq = []
    xs = []
    x0,y0,z0,vx0,vy0,vz0 = sp.symbols('x0 y0 z0 vx0 vy0 vz0')
    xs.extend([x0,y0,z0,vx0,vy0,vz0])
    g = []
    s = I()
    while s:
        v = s.replace('@',',').replace(' ','').split(',')
        v = [int(x) for x in v]
        g.append(v)
        s = I()
    for i,(x,y,z,vx,vy,vz) in enumerate(g):
        t = sp.symbols(f't{i}')
        eq.append(x0 + vx0 * t - x - vx * t)
        eq.append(y0 + vy0 * t - y - vy * t)
        eq.append(z0 + vz0 * t - z - vz * t)
        xs.append(t)
        if len(eq) > len(xs):
            break
    ans = sp.solve(eq, xs, dict=True)
    print(ans)
    print(ans[0][x0] + ans[0][y0] + ans[0][z0])

# day 25
def aoc_25_1():
    s = I()
    g = defaultdict(list)
    while s:
        s = s.replace(':', '')
        v = s.split()
        for nxt in v[1:]:
            g[v[0]].append(nxt)
            g[nxt].append(v[0])
        s = I()
    s = set([('fch','fvh'),('jbz','sqh'),('nvg','vfj')])
    #s = set([('hfx','pzl'),('bvb','cmg'),('nvd','jqt')])
    
    
    def dfs(cur):
        vis = set()
        def f(cur):
            vis.add(cur)
            res = 1
            for nxt in g[cur]:
                if nxt in vis or (nxt, cur) in s or (cur, nxt) in s:
                    continue
                else:
                    res += f(nxt)
            return res
        return f(cur)
    print(dfs('fch') * dfs('fvh'))   

                
                
    # import networkx as nx
    # import matplotlib.pyplot as plt
    # G = nx.Graph()
    # es = 0
    # for k, v in g.items():
    #     G.add_node(k)
    #     for i in v:
    #         G.add_edge(k, i)
    #         es += 1
    # print(es)
        

    
    # nx.draw(G, with_labels=True,node_shape='o',node_color='b',node_size=10)
    # plt.show()


def slove():

    aoc_25_1()

    return


    
T = 1
for _ in range(T):
    slove()



        

# end-------------------------------------------------------
