import time

local = time.localtime(time.time())
print(time.time())
print(time.strftime("%Y-%m-%d %H:%M:%S", local))
print(time.asctime(local))