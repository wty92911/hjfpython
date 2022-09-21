# 由于多进程在Jupyter Notebook中可能无法运行，这里采用魔法方法writefile将此代码块保存为本地py文件

import time
from multiprocessing import Process, Pipe

def pipe_reader(pipe):
    with open('trade.log') as f:
        content = f.read()
    pipe.send(content)
    pipe.close()
    #raise NotImplementedError # TODO: 补全利用pipe发送信息部分

def pipe_writer(pipe):
    content = pipe.recv()
    with open('trade_with_time.log','w') as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  + '\n')
        f.write(content)
    
    #raise NotImplementedError # TODO: 从pipe的另一端接收信息，在文件开头添加时间，并写入trade_with_time.log
    
if __name__ == '__main__':
    con1,con2 = Pipe()
    p1 = Process(target = pipe_reader,args = (con1,))
    p2 = Process(target = pipe_writer,args = (con2,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    #raise NotImplementedError # 创建Pipe，创建并执行两个子进程
