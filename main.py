from tools.text import cosSim

a = cosSim()
r = a.CalcuSim(["你好奥众，今天是星期三", "你好奥迪，今天是星期五"])
print(r)
r = a.CalcuSim(["你好奥众，今天是星期三", "你好大众，今天是星期四"])
print(r)

