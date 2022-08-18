from tkinter import *
from login import *
import tkinter as tk

root = tk.Tk()
root.title('欢迎进入学生成绩管理系统')
LoginPage(root)
root.mainloop()  #使其一直循环不然窗口仅仅占用一个消息的时间,我们肉眼是无法观察到的
