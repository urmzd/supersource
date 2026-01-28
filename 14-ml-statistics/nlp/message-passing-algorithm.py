#!/usr/bin/env python
from abc import ABC, abstractmethod
from typing import Optional


InnerFTable = dict[str, float]
FTable = dict[str, InnerFTable]
Words = list[str]

def parse_stdin(path: str) -> tuple[Words, InnerFTable, FTable, FTable]:
    with open(path, "r") as file:
        words = []
        f1 = {}
        f2 = {}
        f3 = {}

        lines = file.readlines()

        headers = ["#content", "#f1", "#f2", "#f3"]

        i = 0
        mode = 0
        while i < len(lines):
            # Read in line.
            line = lines[i].strip()

            # Acknowledge read.
            i += 1

            # Determine input type.
            if not line:
                mode = 0
            else:
                if not mode:
                    try:
                        mode = headers.index(line) + 1
                        continue
                    except:
                        raise Exception("Something went deeply wrong...")

            if mode:
                # Convert CSV values to list
                terminals = [word.strip() for word in line.split(",")]

                # Logic handlers.
                if mode == 1:
                    words = terminals[0].split(" ")
                elif mode == 2:
                    t, pr = terminals
                    f1[t] = float(pr)
                elif mode == 3:
                    t, w, pr = terminals
                    t_dict = f2.setdefault(t, {})
                    t_dict[w] = float(pr)
                else:
                    t, t2, pr = terminals
                    t_dict = f3.setdefault(t, {})
                    t_dict[t2] = float(pr)

        return words, f1, f2, f3


"""
    @todo: Generalize algorthm
"""
class Node(ABC):
    def __init__(
        self,
        value: Optional[str],
        inputs: Optional[list["Node"]],
        outputs: Optional[list["Node"]],
        data: dict[str, float] = {},
    ) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.data = data
        self.value = value

    @abstractmethod
    def calculate():
        raise Exception("NOT IMPLEMENTED.")


class Message:
    pass


class Factor(Node):
    pass


class Variable(Node):
    pass


def printr(title: str, *d: dict):
    for _d in d:
        print(f"<--------- {title} ---------->")
        for k, v in _d.items():
            print(f"{k:<20}{v:<20}")


print(f"<--------- Part B ---------->")
words, f1, f2, f3 = parse_stdin("input.txt")
# Rule #1 -> (Single) No-N-F -> V
m1 = {k: v for k, v in f1.items()}
printr("M1", m1)

# Rule #2, V=x
m2 = {k: 1.0 if k == words[0] else 0.0 for k in words}
printr("M2", m2)

# Rule #3, V -> Factor -> V
m3 = {t: max([m2[w] * f2[t][w] for w in words]) for t in f1}
printr("M3", m3)

# Rule #1
m4 = {k: m3[k] * m1[k] for k in m1}
printr("M4", m4)

# Rule #3
m5 = {k: max([f3[k][k2] * m4[k] for k2 in f1]) for k in f1}
printr("M5", m5)

# Rule #2
m6 = {k: 1.0 if k == words[1] else 0.0 for k in words}
printr("M6", m6)

# Rule #3, V -> Factor -> V
m7 = {t: max([m6[w] * f2[t][w] for w in words]) for t in f1}
printr("M7", m7)

# Rule #1
m8 = {k: m7[k] * m5[k] for k in m5}
printr("M8", m8)


# Rule #3
m9 = {k: max([f3[k][k2] * m8[k] for k2 in f1]) for k in f1}
printr("M9", m9)

# Rule #2, V=x
m10 = {k: 1.0 if k == words[2] else 0.0 for k in words}
printr("M10", m10)

# Rule #3, V -> Factor -> V
m11 = {t: max([m10[w] * f2[t][w] for w in words]) for t in f1}
printr("M11", m11)

# Rule #1
m12 = {k: m11[k] * m9[k] for k in m1}
printr("M12", m12)

# Rule #3
m13 = {k: max([f3[k][k2] * m12[k] for k2 in f1]) for k in f1}
printr("M13", m13)

# Rule #2, V=x
m14 = {k: 1.0 if k == words[3] else 0.0 for k in words}
printr("M14", m14)

# Rule #3, V -> Factor -> V
m15 = {t: max([m14[w] * f2[t][w] for w in words]) for t in f1}
printr("M15", m15)

# Rule #2
t4 = {k: m13[k] * m15[k] for k in m13}
t4_max = max(t4, key=lambda k: t4[k])
t4a = {k: t4[t4_max] if k == t4_max else 0. for k in t4}

# F -> V -> F
m16 = {k: t4a[k] for k in t4a}
printr("M16", m16)

# Rule #3
m17 = {k: t4a[t4_max] * f3[t4_max][k] for k in f3[t4_max]}
printr("M17", m17)

# Rule #3
t3 = {k: m17[k] * m9[k] * m11[k] for k in m17}
t3_max = max(t3, key=lambda x: t3[x])
t3a =  {k: t3[t3_max] if k == t3_max else 0. for k in t3}

m18 = {k: m17[k] * m11[k] for k in m17}
printr("M18", m18)

m19 = {k: m18[t3_max] * f3[t3_max][k] for k in f3[t3_max]}
printr("M19", m19)

t2 = {k: m19[k] * m5[k] * m7[k] for k in m19}
t2_max = max(t2, key=lambda x: t2[x])
t2a = {k: t2[t2_max] if k == t2_max else 0. for k in t2}

m20 = {k: m7[k] * m19[k] for k in m7}
printr("M20", m20)

m21 = {k: m20[t2_max] * f3[t3_max][k] for k in f3[t2_max]}
printr("M21", m21)

t1 = {k: m21[k] * m3[k] * m1[k] for k in m21}
t1_max = max(t1, key=lambda x: t1[x])
t1a = {k: t1[t1_max] if k == t1_max else 0. for k in t1}


print(f"<--------- Part C ---------->")
print(f"T1 = {t1_max.upper()}, T2 = {t2_max.upper()}, T3 = {t3_max.upper()}, T4 = {t4_max.upper()}")
