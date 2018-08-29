from tools.text import cosSim

a = cosSim()
r = a.CalcuSim(["你好奥众，今天是星期三", "你好奥迪，今天是星期五"])
print(r)
r = a.CalcuSim(["你好奥众，今天是星期三", "你好大众，今天是星期四"])
print(r)

r = a.CalcuSim(["以下是README的一些说明，我们可以看到模型的识别效果总体还是不错的", "我们可以看到模型的识别效果总体还是不错的，以下是README的一些说明"])
print(r)

