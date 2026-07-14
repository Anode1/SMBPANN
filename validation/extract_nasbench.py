import re, sys, time
from nats_bench import create
OPS=['none','skip_connect','nor_conv_1x1','nor_conv_3x3','avg_pool_3x3']
oi={o:i for i,o in enumerate(OPS)}
t0=time.time()
print("loading pickle (this decompresses ~1GB)...", flush=True)
api=create('/tmp/claude-1000/nats_tss.pickle.pbz2','tss',fast_mode=False,verbose=False)
print("loaded %d archs in %.0fs" % (len(api), time.time()-t0), flush=True)
def parse(s): return [oi[o] for o in re.findall(r'([a-z0-9_]+)~\d', s)]
dss=['cifar10','cifar100','ImageNet16-120']
with open('/tmp/claude-1000/nasbench201.txt','w') as f:
    f.write("# op0 op1 op2 op3 op4 op5  cifar10 cifar100 imagenet16  (test-accuracy, 200ep)\n")
    for i in range(len(api)):
        tup=parse(api.arch(i))
        accs=[api.get_more_info(i,ds,hp='200',is_random=False)['test-accuracy'] for ds in dss]
        f.write(' '.join(map(str,tup))+' '+' '.join('%.4f'%a for a in accs)+'\n')
        if i%2000==0: print("  %d/%d" % (i,len(api)), flush=True)
print("done, wrote nasbench201.txt in %.0fs" % (time.time()-t0), flush=True)
