#!/usr/bin/python3

import sys, time, random

events = {"END", "EPOCH", "MINE", "TRX", "BLK"}
count = 0
if sys.argv[1] in events:
  threshold = sys.argv[1]
else:
  threshold = float(sys.argv[1])
  if threshold >= 2:
    count = int(threshold) + 1
    threshold = 0

for line in sys.stdin.readlines():
  inp = line.split()
  if len(inp) > 0 and inp[0] in events:
    if threshold in events:
      if threshold == inp[0]:
        time.sleep(0.1)
    elif count == 1:
      time.sleep(0.1)
    elif random.random() < threshold:
      time.sleep(0.1)

  print(line,end="",flush=True)
  count -= 1
  
exit(0)

for line in sys.stdin.readlines():
  inp = line.split()
  if len(inp) > 0 and inp[0] == "#":
    time.sleep(0.1)
  else:
    print(line,end="",flush=True)
