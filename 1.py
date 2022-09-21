news=[]
with open('news.txt','r',encoding='utf-8') as f:
    news=f.readlines()
import nest_asyncio
import asyncio,time
import re
a = '.{0,}全国.{0,}委员会.{0,}'
b = '.{0,}中国.{0,}协会.{0,}'
c = '.{0,}中美建交.{0,}'
nest_asyncio.apply() 

async def consumer(q):
    print('consumer starts.')
    #TODO
    while True:
        x = await q.get()
        #print(x)
        if x is None:
            q.task_done()
            break
        else:
            #print(x)
            if re.match(a,x) and re.match(b,x) and re.match(c,x):
                print(x)
              #  pass
            q.task_done()
    print("consumer ends.")
    
async def producer(q):
    print('producer starts.')
    #TODO
    for x in news:
        await q.put(x)
    await q.put(None)
    await q.join()
    print("producer ends.")
    
q=asyncio.Queue(maxsize=10)
t0=time.time()
loop=asyncio.get_event_loop()
tasks=[producer(q),consumer(q)]
loop.run_until_complete(asyncio.wait(tasks))
loop.close()
print(time.time()-t0,"s")