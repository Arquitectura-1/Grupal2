import os
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import time

outRoute = "/home/bryan"
benchRoute = "/home/bryan/Downloads/hw2-benchmarks/benchmarks"
sjengRoute = "/home/bryan/Project1_SPEC/458.sjeng"
'''
#CPU 0, Base
os.system("echo benhmarks cpu1")
os.system("sudo sudo timeout --signal=SIGINT 110s ./gem5/build/X86/gem5.opt -d "+outRoute+"/cpu0/b1 ./gem5/configs/example/se.py -c "+sjengRoute+"/src/benchmark --cpu-type=AtomicSimpleCPU --caches --l2cache --l1d_size=1kB --l1i_size=1kB --l2_size=1kB --l1d_assoc=2 --l1i_assoc=2 --l2_assoc=2 --cacheline_size=64 --l2-hwp-type=SignaturePathPrefetcher --bp-type=BiModeBP")
os.system("sudo ./gem5/build/X86/gem5.opt -d "+outRoute+"/cpu0/b2 ./gem5/configs/example/se.py -c "+benchRoute+"/BFS --options="+ "\"-o test.txt -r 10 "+benchRoute+"/inputs/RL3k.graph\" " +"--cpu-type=AtomicSimpleCPU --caches --l2cache --l1d_size=128kB --l1i_size=1kB --l2_size=1kB --l1d_assoc=2 --l1i_assoc=2 --l2_assoc=2 --cacheline_size=64 --l2-hwp-type=SignaturePathPrefetcher --bp-type=BiModeBP")
os.system("sudo ./gem5/build/X86/gem5.opt -d "+outRoute+"/cpu0/b3 ./gem5/configs/example/se.py -c "+benchRoute+"/blocked-matmul --cpu-type=AtomicSimpleCPU --caches --l2cache --l1d_size=128kB --l1i_size=1kB --l2_size=1kB --l1d_assoc=2 --l1i_assoc=2 --l2_assoc=2 --cacheline_size=64 --l2-hwp-type=SignaturePathPrefetcher --bp-type=BiModeBP")
os.system("sudo ./gem5/build/X86/gem5.opt -d "+outRoute+"/cpu0/b4 ./gem5/configs/example/se.py -c "+benchRoute+"/sha --cpu-type=AtomicSimpleCPU --caches --l2cache --l1d_size=128kB --l1i_size=1kB --l2_size=1kB --l1d_assoc=2 --l1i_assoc=2 --l2_assoc=2 --cacheline_size=64 --l2-hwp-type=SignaturePathPrefetcher --bp-type=BiModeBP")
os.system("sudo ./gem5/build/X86/gem5.opt -d "+outRoute+"/cpu0/b5 ./gem5/configs/example/se.py -c "+benchRoute+"/queens --options="+"\"-c 10\" --cpu-type=AtomicSimpleCPU --caches --l2cache --l1d_size=128kB --l1i_size=1kB --l2_size=1kB --l1d_assoc=2 --l1i_assoc=2 --l2_assoc=2 --cacheline_size=64 --l2-hwp-type=SignaturePathPrefetcher --bp-type=BiModeBP")

#CPU 1, implementa prefetcher
os.system("echo benhmarks cpu1")
os.system("sudo sudo timeout --signal=SIGINT 110s ./gem5/build/X86/gem5.opt -d "+outRoute+"/cpu1/b1 ./gem5/configs/example/se.py -c "+sjengRoute+"/src/benchmark --cpu-type=AtomicSimpleCPU --caches --l2cache --l1d_size=128kB --l1i_size=1kB --l2_size=1kB --l1d_assoc=2 --l1i_assoc=2 --l2_assoc=2 --cacheline_size=64 --l2-hwp-type=BOPPrefetcher --bp-type=BiModeBP")
os.system("sudo ./gem5/build/X86/gem5.opt -d "+outRoute+"/cpu1/b2 ./gem5/configs/example/se.py -c "+benchRoute+"/BFS --options="+ "\"-o test.txt -r 10 "+benchRoute+"/inputs/RL3k.graph\" " +"--cpu-type=AtomicSimpleCPU --caches --l2cache --l1d_size=128kB --l1i_size=1kB --l2_size=1kB --l1d_assoc=2 --l1i_assoc=2 --l2_assoc=2 --cacheline_size=64 --l2-hwp-type=BOPPrefetcher --bp-type=BiModeBP")
os.system("sudo ./gem5/build/X86/gem5.opt -d "+outRoute+"/cpu1/b3 ./gem5/configs/example/se.py -c "+benchRoute+"/blocked-matmul --cpu-type=AtomicSimpleCPU --caches --l2cache --l1d_size=128kB --l1i_size=1kB --l2_size=1kB --l1d_assoc=2 --l1i_assoc=2 --l2_assoc=2 --cacheline_size=64 --l2-hwp-type=BOPPrefetcher --bp-type=BiModeBP")
os.system("sudo ./gem5/build/X86/gem5.opt -d "+outRoute+"/cpu1/b4 ./gem5/configs/example/se.py -c "+benchRoute+"/sha --cpu-type=AtomicSimpleCPU --caches --l2cache --l1d_size=128kB --l1i_size=1kB --l2_size=1kB --l1d_assoc=2 --l1i_assoc=2 --l2_assoc=2 --cacheline_size=64 --l2-hwp-type=BOPPrefetcher --bp-type=BiModeBP")
os.system("sudo ./gem5/build/X86/gem5.opt -d "+outRoute+"/cpu1/b5 ./gem5/configs/example/se.py -c "+benchRoute+"/queens --options="+"\"-c 10\" --cpu-type=AtomicSimpleCPU --caches --l2cache --l1d_size=128kB --l1i_size=1kB --l2_size=1kB --l1d_assoc=2 --l1i_assoc=2 --l2_assoc=2 --cacheline_size=64 --l2-hwp-type=BOPPrefetcher --bp-type=BiModeBP")

#CPU 2, Implementa politica de reemplazo
os.system("echo benhmarks cpu2")
os.system("sudo sudo timeout --signal=SIGINT 110s ./gem5/build/X86/gem5.opt -d "+outRoute+"/cpu2/b1 ./gem5/configs/example/se.py -c "+sjengRoute+"/src/benchmark --cpu-type=AtomicSimpleCPU --caches --l2cache --l1d_size=128kB --l1i_size=1kB --l2_size=1kB --l1d_assoc=2 --l1i_assoc=2 --l2_assoc=2 --cacheline_size=64 --l2_poli=\"WeightedLRURP()\" --l1d_poli=\"WeightedLRURP()\" --l1i_poli=\"WeightedLRURP()\" --l2-hwp-type=SignaturePathPrefetcher --bp-type=BiModeBP")
os.system("sudo ./gem5/build/X86/gem5.opt -d "+outRoute+"/cpu2/b2 ./gem5/configs/example/se.py -c "+benchRoute+"/BFS --options="+ "\"-o test.txt -r 10 "+benchRoute+"/inputs/RL3k.graph\" " +"--cpu-type=AtomicSimpleCPU --caches --l2cache --l1d_size=128kB --l1i_size=1kB --l2_size=1kB --l1d_assoc=2 --l1i_assoc=2 --l2_assoc=2 --cacheline_size=64 --l2_poli=\"WeightedLRURP()\" --l1d_poli=\"WeightedLRURP()\" --l1i_poli=\"WeightedLRURP()\" --l2-hwp-type=SignaturePathPrefetcher --bp-type=BiModeBP")
os.system("sudo ./gem5/build/X86/gem5.opt -d "+outRoute+"/cpu2/b3 ./gem5/configs/example/se.py -c "+benchRoute+"/blocked-matmul --cpu-type=AtomicSimpleCPU --caches --l2cache --l1d_size=128kB --l1i_size=1kB --l2_size=1kB --l1d_assoc=2 --l1i_assoc=2 --l2_assoc=2 --cacheline_size=64 --l2_poli=\"WeightedLRURP()\" --l1d_poli=\"WeightedLRURP()\" --l1i_poli=\"WeightedLRURP()\" --l2-hwp-type=SignaturePathPrefetcher --bp-type=BiModeBP")
os.system("sudo ./gem5/build/X86/gem5.opt -d "+outRoute+"/cpu2/b4 ./gem5/configs/example/se.py -c "+benchRoute+"/sha --cpu-type=AtomicSimpleCPU --caches --l2cache --l1d_size=128kB --l1i_size=1kB --l2_size=1kB --l1d_assoc=2 --l1i_assoc=2 --l2_assoc=2 --cacheline_size=64 --l2_poli=\"WeightedLRURP()\" --l1d_poli=\"WeightedLRURP()\" --l1i_poli=\"WeightedLRURP()\" --l2-hwp-type=SignaturePathPrefetcher --bp-type=BiModeBP")
os.system("sudo ./gem5/build/X86/gem5.opt -d "+outRoute+"/cpu2/b5 ./gem5/configs/example/se.py -c "+benchRoute+"/queens --options="+"\"-c 10\" --cpu-type=AtomicSimpleCPU --caches --l2cache --l1d_size=128kB --l1i_size=1kB --l2_size=1kB --l1d_assoc=2 --l1i_assoc=2 --l2_assoc=2 --cacheline_size=64 --l2_poli=\"WeightedLRURP()\" --l1d_poli=\"WeightedLRURP()\" --l1i_poli=\"WeightedLRURP()\" --l2-hwp-type=SignaturePathPrefetcher --bp-type=BiModeBP")

#CPU 3, Latencia
os.system("echo benhmarks cpu3")
os.system("sudo sudo timeout --signal=SIGINT 110s ./gem5/build/X86/gem5.opt -d "+outRoute+"/cpu3/b1 ./gem5/configs/example/se.py -c "+sjengRoute+"/src/benchmark --cpu-type=AtomicSimpleCPU --caches --l2cache --l1d_size=128kB --l1i_size=1kB --l2_size=1kB --l1d_assoc=2 --l1i_assoc=2 --l2_assoc=2 --cacheline_size=64 --l2_rlatency=10 --l1d_rlatency=10 --l1i_rlatency=10 --l2-hwp-type=SignaturePathPrefetcher --bp-type=BiModeBP")
os.system("sudo ./gem5/build/X86/gem5.opt -d "+outRoute+"/cpu3/b2 ./gem5/configs/example/se.py -c "+benchRoute+"/BFS --options="+ "\"-o test.txt -r 10 "+benchRoute+"/inputs/RL3k.graph\" " +"--cpu-type=AtomicSimpleCPU --caches --l2cache --l1d_size=128kB --l1i_size=1kB --l2_size=1kB --l1d_assoc=2 --l1i_assoc=2 --l2_assoc=2 --cacheline_size=64 --l2_rlatency=10 --l1d_rlatency=10 --l1i_rlatency=10 --l2-hwp-type=SignaturePathPrefetcher --bp-type=BiModeBP")
os.system("sudo ./gem5/build/X86/gem5.opt -d "+outRoute+"/cpu3/b3 ./gem5/configs/example/se.py -c "+benchRoute+"/blocked-matmul --cpu-type=AtomicSimpleCPU --caches --l2cache --l1d_size=128kB --l1i_size=1kB --l2_size=1kB --l1d_assoc=2 --l1i_assoc=2 --l2_assoc=2 --cacheline_size=64 --l2_rlatency=10 --l1d_rlatency=10 --l1i_rlatency=10 --l2-hwp-type=SignaturePathPrefetcher --bp-type=BiModeBP")
os.system("sudo ./gem5/build/X86/gem5.opt -d "+outRoute+"/cpu3/b4 ./gem5/configs/example/se.py -c "+benchRoute+"/sha --cpu-type=AtomicSimpleCPU --caches --l2cache --l1d_size=128kB --l1i_size=1kB --l2_size=1kB --l1d_assoc=2 --l1i_assoc=2 --l2_assoc=2 --cacheline_size=64 --l2_rlatency=10 --l1d_rlatency=10 --l1i_rlatency=10 --l2-hwp-type=SignaturePathPrefetcher --bp-type=BiModeBP")
os.system("sudo ./gem5/build/X86/gem5.opt -d "+outRoute+"/cpu3/b5 ./gem5/configs/example/se.py -c "+benchRoute+"/queens --options="+"\"-c 10\" --cpu-type=AtomicSimpleCPU --caches --l2cache --l1d_size=128kB --l1i_size=1kB --l2_size=1kB --l1d_assoc=2 --l1i_assoc=2 --l2_assoc=2 --cacheline_size=64 --l2_rlatency=10 --l1d_rlatency=10 --l1i_rlatency=10 --l2-hwp-type=SignaturePathPrefetcher --bp-type=BiModeBP")

#CPU 4, Incrementa el grado de asociatividad
os.system("echo benhmarks cpu4")
os.system("sudo sudo timeout --signal=SIGINT 110s ./gem5/build/X86/gem5.opt -d "+outRoute+"/cpu4/b1 ./gem5/configs/example/se.py -c "+sjengRoute+"/src/benchmark --cpu-type=AtomicSimpleCPU --caches --l2cache --l1d_size=128kB --l1i_size=1kB --l2_size=1kB --l1d_assoc=8 --l1i_assoc=8 --l2_assoc=8 --cacheline_size=64 --l2-hwp-type=SignaturePathPrefetcher --bp-type=BiModeBP")
os.system("sudo ./gem5/build/X86/gem5.opt -d "+outRoute+"/cpu4/b2 ./gem5/configs/example/se.py -c "+benchRoute+"/BFS --options="+ "\"-o test.txt -r 10 "+benchRoute+"/inputs/RL3k.graph\" " +"--cpu-type=AtomicSimpleCPU --caches --l2cache --l1d_size=128kB --l1i_size=1kB --l2_size=1kB --l1d_assoc=8 --l1i_assoc=8 --l2_assoc=8 --cacheline_size=64 --l2-hwp-type=SignaturePathPrefetcher --bp-type=BiModeBP")
os.system("sudo ./gem5/build/X86/gem5.opt -d "+outRoute+"/cpu4/b3 ./gem5/configs/example/se.py -c "+benchRoute+"/blocked-matmul --cpu-type=AtomicSimpleCPU --caches --l2cache --l1d_size=128kB --l1i_size=1kB --l2_size=1kB --l1d_assoc=8 --l1i_assoc=8 --l2_assoc=8 --cacheline_size=64 --l2-hwp-type=SignaturePathPrefetcher --bp-type=BiModeBP")
os.system("sudo ./gem5/build/X86/gem5.opt -d "+outRoute+"/cpu4/b4 ./gem5/configs/example/se.py -c "+benchRoute+"/sha --cpu-type=AtomicSimpleCPU --caches --l2cache --l1d_size=128kB --l1i_size=1kB --l2_size=1kB --l1d_assoc=8 --l1i_assoc=8 --l2_assoc=8 --cacheline_size=64 --l2-hwp-type=SignaturePathPrefetcher --bp-type=BiModeBP")
os.system("sudo ./gem5/build/X86/gem5.opt -d "+outRoute+"/cpu4/b5 ./gem5/configs/example/se.py -c "+benchRoute+"/queens --options="+"\"-c 10\" --cpu-type=AtomicSimpleCPU --caches --l2cache --l1d_size=128kB --l1i_size=1kB --l2_size=1kB --l1d_assoc=8 --l1i_assoc=8 --l2_assoc=8 --cacheline_size=64 --l2-hwp-type=SignaturePathPrefetcher --bp-type=BiModeBP")

#CPU 5, Implementa un branch predictor
os.system("echo benhmarks cpu5")
os.system("sudo sudo timeout --signal=SIGINT 110s ./gem5/build/X86/gem5.opt -d "+outRoute+"/cpu5/b1 ./gem5/configs/example/se.py -c "+sjengRoute+"/src/benchmark --cpu-type=AtomicSimpleCPU --caches --l2cache --l1d_size=128kB --l1i_size=1kB --l2_size=1kB --l1d_assoc=2 --l1i_assoc=2 --l2_assoc=2 --cacheline_size=64 --bp-type=MultiperspectivePerceptronTAGE64KB --l2-hwp-type=SignaturePathPrefetcher")
os.system("sudo ./gem5/build/X86/gem5.opt -d "+outRoute+"/cpu5/b2 ./gem5/configs/example/se.py -c "+benchRoute+"/BFS --options="+ "\"-o test.txt -r 10 "+benchRoute+"/inputs/RL3k.graph\" " +"--cpu-type=AtomicSimpleCPU --caches --l2cache --l1d_size=128kB --l1i_size=1kB --l2_size=1kB --l1d_assoc=2 --l1i_assoc=2 --l2_assoc=2 --cacheline_size=64 --bp-type=MultiperspectivePerceptronTAGE64KB --l2-hwp-type=SignaturePathPrefetcher")
os.system("sudo ./gem5/build/X86/gem5.opt -d "+outRoute+"/cpu5/b3 ./gem5/configs/example/se.py -c "+benchRoute+"/blocked-matmul --cpu-type=AtomicSimpleCPU --caches --l2cache --l1d_size=128kB --l1i_size=1kB --l2_size=1kB --l1d_assoc=2 --l1i_assoc=2 --l2_assoc=2 --cacheline_size=64 --bp-type=MultiperspectivePerceptronTAGE64KB --l2-hwp-type=SignaturePathPrefetcher")
os.system("sudo ./gem5/build/X86/gem5.opt -d "+outRoute+"/cpu5/b4 ./gem5/configs/example/se.py -c "+benchRoute+"/sha --cpu-type=AtomicSimpleCPU --caches --l2cache --l1d_size=128kB --l1i_size=1kB --l2_size=1kB --l1d_assoc=2 --l1i_assoc=2 --l2_assoc=2 --cacheline_size=64 --bp-type=MultiperspectivePerceptronTAGE64KB --l2-hwp-type=SignaturePathPrefetcher")
os.system("sudo ./gem5/build/X86/gem5.opt -d "+outRoute+"/cpu5/b5 ./gem5/configs/example/se.py -c "+benchRoute+"/queens --options="+"\"-c 10\" --cpu-type=AtomicSimpleCPU --caches --l2cache --l1d_size=128kB --l1i_size=1kB --l2_size=1kB --l1d_assoc=2 --l1i_assoc=2 --l2_assoc=2 --cacheline_size=64 --bp-type=MultiperspectivePerceptronTAGE64KB --l2-hwp-type=SignaturePathPrefetcher")


'''
##Apartado de graficas
valores0 =[]
def graph():
    #  PARA BENCHMARK 1     
    muestra = open("/home/bryan/cpu0/b1/stats.txt",'r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                print("caso1B1")
                valores0.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.branchPred.condIncorrect"):
                print("caso2B1")
                valores0.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                print("caso3B1")
                valores0.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                print("caso4B1")
                valores0.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                print("caso5B1")
                valores0.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.num_reads::total"):
                print("caso6B1")
                valores0.append(math.log10(int(columna[1])))

    valores =[]  
    muestra = open(outRoute+"/cpu1/b1/stats.txt",'r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.branchPred.condIncorrect"):
                valores.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.num_reads::total"):
                valores.append(math.log10(int(columna[1])))
            
    valores2 =[]
    muestra = open(outRoute+"/cpu2/b1/stats.txt",'r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores2.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.branchPred.condIncorrect"):
                valores2.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores2.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores2.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores2.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.num_reads::total"):
                valores2.append(math.log10(int(columna[1])))
    
    valores3 =[]
    muestra = open(outRoute+"/cpu3/b1/stats.txt",'r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores3.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.branchPred.condIncorrect"):
                valores3.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores3.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores3.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores3.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.num_reads::total"):
                valores3.append(math.log10(int(columna[1])))

    valores4 =[]
    muestra = open(outRoute+"/cpu4/b1/stats.txt",'r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores4.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.branchPred.condIncorrect"):
                valores4.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores4.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores4.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores4.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.num_reads::total"):
                valores4.append(math.log10(int(columna[1])))
    
    valores5 =[]
    muestra = open(outRoute+"/cpu5/b1/stats.txt",'r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores5.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.branchPred.condIncorrect"):
                valores5.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores5.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores5.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores5.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.num_reads::total"):
                valores5.append(math.log10(int(columna[1])))
    
    #fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(100,100))
    #plt.subplots_adjust(left=0.03, bottom=0.327, right=0.977, hspace=0.88)

    df = pd.DataFrame([['CPU0','BranchMisses.',valores0[1]],['CPU0','CL1 d misses.',valores0[2]],
                       ['CPU0','CL1 i misses.',valores0[3]],['CPU0','CL2 misses',valores0[4]],['CPU0','RAM reads',valores0[5]],
                       ['CPU1','BranchMisses.',valores[1]],['CPU1','CL1 d misses.',valores[2]],
                       ['CPU1','CL1 i misses.',valores[3]],['CPU1','CL2 misses',valores[4]],['CPU1','RAM reads',valores[5]],
                       ['CPU2','BranchMisses.',valores2[1]],['CPU2','CL1 d misses.',valores2[2]],
                       ['CPU2','CL1 i misses.',valores2[3]], ['CPU2','CL2 misses',valores2[4]],['CPU2','RAM reads',valores2[5]],
                       ['CPU3','BranchMisses.',valores3[1]],['CPU3','CL1 d misses.',valores3[2]],
                       ['CPU3','CL1 i misses.',valores3[3]],['CPU3','CL2 misses',valores3[4]],['CPU3','RAM reads',valores3[5]],
                       ['CPU4','BranchMisses.',valores4[1]],['CPU4','CL1 d misses.',valores4[2]],
                       ['CPU4','CL1 i misses.',valores4[3]],['CPU4','CL2 misses',valores4[4]],['CPU4','RAM reads',valores4[5]],
                       ['CPU5','BranchMisses.',valores5[1]],['CPU5','CL1 d misses.',valores5[2]],
                       ['CPU5','CL1 i misses.',valores5[3]],['CPU5','CL2 misses',valores5[4]],['CPU5','RAM reads',valores5[5]],
                       ], columns=['CPU','Sjeng','val'])

    df.pivot("Sjeng", "CPU", "val").plot(kind='barh')

    #  PARA BENCHMARK 1
    valores20 = []
    muestra = open(outRoute+"/cpu0/b2/stats.txt",'r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores20.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.branchPred.condIncorrect"):
                valores20.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores20.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores20.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores20.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.num_reads::total"):
                valores20.append(math.log10(int(columna[1])))

    valores21 = []
    muestra = open(outRoute+"/cpu1/b2/stats.txt",'r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores21.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.branchPred.condIncorrect"):
                valores21.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores21.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores21.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores21.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.num_reads::total"):
                valores21.append(math.log10(int(columna[1])))
            
    valores22 =[]
    muestra = open(outRoute+"/cpu2/b2/stats.txt",'r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores22.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.branchPred.condIncorrect"):
                valores22.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores22.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores22.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores22.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.num_reads::total"):
                valores22.append(math.log10(int(columna[1])))
    
    valores23 =[]
    muestra = open(outRoute+"/cpu3/b2/stats.txt",'r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores23.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.branchPred.condIncorrect"):
                valores23.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores23.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores23.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores23.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.num_reads::total"):
                valores23.append(math.log10(int(columna[1])))

    valores24 =[]
    muestra = open(outRoute+"/cpu4/b2/stats.txt",'r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores24.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.branchPred.condIncorrect"):
                valores24.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores24.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores24.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores24.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.num_reads::total"):
                valores24.append(math.log10(int(columna[1])))
    
    valores25 =[]
    muestra = open(outRoute+"/cpu5/b2/stats.txt",'r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores25.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.branchPred.condIncorrect"):
                valores25.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores25.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores25.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores25.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.num_reads::total"):
                valores25.append(math.log10(int(columna[1])))
    

    df2 = pd.DataFrame([['CPU0','BranchMisses.',valores20[1]],['CPU0','CL1 d misses.',valores20[2]],
                       ['CPU0','CL1 i misses.',valores20[3]],['CPU0','CL2 misses',valores20[4]],['CPU0','RAM reads',valores20[5]],
                       ['CPU1','BranchMisses.',valores21[1]],['CPU1','CL1 d misses.',valores21[2]],
                       ['CPU1','CL1 i misses.',valores21[3]],['CPU1','CL2 misses',valores21[4]],['CPU1','RAM reads',valores21[5]],
                       ['CPU2','BranchMisses.',valores22[1]],['CPU2','CL1 d misses.',valores22[2]],
                       ['CPU2','CL1 i misses.',valores22[3]], ['CPU2','CL2 misses',valores22[4]],['CPU2','RAM reads',valores22[5]],
                       ['CPU3','BranchMisses.',valores23[1]],['CPU3','CL1 d misses.',valores23[2]],
                       ['CPU3','CL1 i misses.',valores23[3]],['CPU3','CL2 misses',valores23[4]],['CPU3','RAM reads',valores23[5]],
                       ['CPU4','BranchMisses.',valores24[1]],['CPU4','CL1 d misses.',valores24[2]],
                       ['CPU4','CL1 i misses.',valores24[3]],['CPU4','CL2 misses',valores24[4]],['CPU4','RAM reads',valores24[5]],
                       ['CPU5','BranchMisses.',valores25[1]],['CPU5','CL1 d misses.',valores25[2]],
                       ['CPU5','CL1 i misses.',valores25[3]],['CPU5','CL2 misses',valores25[4]],['CPU5','RAM reads',valores25[5]],
                       ], columns=['CPU','BFS','val'])

    df2.pivot("BFS", "CPU", "val").plot(kind='barh')

    #  PARA BENCHMARK 3
    valores30 = []
    muestra = open(outRoute+"/cpu0/b3/stats.txt",'r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores30.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.branchPred.condIncorrect"):
                valores30.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores30.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores30.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores30.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.num_reads::total"):
                valores30.append(math.log10(int(columna[1])))

    valores31 = []
    muestra = open(outRoute+"/cpu1/b3/stats.txt",'r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores31.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.branchPred.condIncorrect"):
                valores31.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores31.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores31.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores31.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.num_reads::total"):
                valores31.append(math.log10(int(columna[1])))
            
    valores32 =[]
    muestra = open(outRoute+"/cpu2/b3/stats.txt",'r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores32.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.branchPred.condIncorrect"):
                valores32.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores32.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores32.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores32.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.num_reads::total"):
                valores32.append(math.log10(int(columna[1])))
    
    valores33 =[]
    muestra = open(outRoute+"/cpu3/b3/stats.txt",'r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores33.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.branchPred.condIncorrect"):
                valores33.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores33.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores33.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores33.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.num_reads::total"):
                valores33.append(math.log10(int(columna[1])))

    valores34 =[]
    muestra = open(outRoute+"/cpu4/b3/stats.txt",'r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores34.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.branchPred.condIncorrect"):
                valores34.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores34.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores34.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores34.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.num_reads::total"):
                valores34.append(math.log10(int(columna[1])))
    
    valores35 =[]
    muestra = open(outRoute+"/cpu5/b3/stats.txt",'r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores35.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.branchPred.condIncorrect"):
                valores35.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores35.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores35.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores35.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.num_reads::total"):
                valores35.append(math.log10(int(columna[1])))
    

    df3 = pd.DataFrame([['CPU0','BranchMisses.',valores30[1]],['CPU0','CL1 d misses.',valores30[2]],
                       ['CPU0','CL1 i misses.',valores30[3]],['CPU0','CL2 misses',valores30[4]],['CPU0','RAM reads',valores30[5]],
                       ['CPU1','BranchMisses.',valores31[1]],['CPU1','CL1 d misses.',valores31[2]],
                       ['CPU1','CL1 i misses.',valores31[3]],['CPU1','CL2 misses',valores31[4]],['CPU1','RAM reads',valores31[5]],
                       ['CPU2','BranchMisses.',valores32[1]],['CPU2','CL1 d misses.',valores32[2]],
                       ['CPU2','CL1 i misses.',valores32[3]], ['CPU2','CL2 misses',valores32[4]],['CPU2','RAM reads',valores32[5]],
                       ['CPU3','BranchMisses.',valores33[1]],['CPU3','CL1 d misses.',valores33[2]],
                       ['CPU3','CL1 i misses.',valores33[3]],['CPU3','CL2 misses',valores33[4]],['CPU3','RAM reads',valores33[5]],
                       ['CPU4','BranchMisses.',valores34[1]],['CPU4','CL1 d misses.',valores34[2]],
                       ['CPU4','CL1 i misses.',valores34[3]],['CPU4','CL2 misses',valores34[4]],['CPU4','RAM reads',valores34[5]],
                       ['CPU5','BranchMisses.',valores35[1]],['CPU5','CL1 d misses.',valores35[2]],
                       ['CPU5','CL1 i misses.',valores35[3]],['CPU5','CL2 misses',valores35[4]],['CPU5','RAM reads',valores35[5]],
                       ], columns=['CPU','BlockMul','val'])

    df3.pivot("BlockMul", "CPU", "val").plot(kind='barh')


    #  PARA BENCHMARK 4
    valores40 = []
    muestra = open(outRoute+"/cpu0/b4/stats.txt",'r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores40.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.branchPred.condIncorrect"):
                valores40.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores40.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores40.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores40.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.num_reads::total"):
                valores40.append(math.log10(int(columna[1])))

    valores41 = []
    muestra = open(outRoute+"/cpu1/b4/stats.txt",'r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores41.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.branchPred.condIncorrect"):
                valores41.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores41.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores41.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores41.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.num_reads::total"):
                valores41.append(math.log10(int(columna[1])))
            
    valores42 =[]
    muestra = open(outRoute+"/cpu2/b4/stats.txt",'r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores42.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.branchPred.condIncorrect"):
                valores42.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores42.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores42.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores42.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.num_reads::total"):
                valores42.append(math.log10(int(columna[1])))
    
    valores43 =[]
    muestra = open(outRoute+"/cpu3/b4/stats.txt",'r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores43.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.branchPred.condIncorrect"):
                valores43.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores43.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores43.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores43.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.num_reads::total"):
                valores43.append(math.log10(int(columna[1])))

    valores44 =[]
    muestra = open(outRoute+"/cpu4/b4/stats.txt",'r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores44.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.branchPred.condIncorrect"):
                valores44.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores44.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores44.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores44.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.num_reads::total"):
                valores44.append(math.log10(int(columna[1])))
    
    valores45 =[]
    muestra = open(outRoute+"/cpu5/b4/stats.txt",'r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores45.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.branchPred.condIncorrect"):
                valores45.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores45.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores45.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores45.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.num_reads::total"):
                valores45.append(math.log10(int(columna[1])))
    

    df4 = pd.DataFrame([['CPU0','BranchMisses.',valores40[1]],['CPU0','CL1 d misses.',valores40[2]],
                       ['CPU0','CL1 i misses.',valores40[3]],['CPU0','CL2 misses',valores40[4]],['CPU0','RAM reads',valores40[5]],
                       ['CPU1','BranchMisses.',valores41[1]],['CPU1','CL1 d misses.',valores41[2]],
                       ['CPU1','CL1 i misses.',valores41[3]],['CPU1','CL2 misses',valores41[4]],['CPU1','RAM reads',valores41[5]],
                       ['CPU2','BranchMisses.',valores42[1]],['CPU2','CL1 d misses.',valores42[2]],
                       ['CPU2','CL1 i misses.',valores42[3]], ['CPU2','CL2 misses',valores42[4]],['CPU2','RAM reads',valores42[5]],
                       ['CPU3','BranchMisses.',valores43[1]],['CPU3','CL1 d misses.',valores43[2]],
                       ['CPU3','CL1 i misses.',valores43[3]],['CPU3','CL2 misses',valores43[4]],['CPU3','RAM reads',valores43[5]],
                       ['CPU4','BranchMisses.',valores44[1]],['CPU4','CL1 d misses.',valores44[2]],
                       ['CPU4','CL1 i misses.',valores44[3]],['CPU4','CL2 misses',valores44[4]],['CPU4','RAM reads',valores44[5]],
                       ['CPU5','BranchMisses.',valores45[1]],['CPU5','CL1 d misses.',valores45[2]],
                       ['CPU5','CL1 i misses.',valores45[3]],['CPU5','CL2 misses',valores45[4]],['CPU5','RAM reads',valores45[5]],
                       ], columns=['CPU','SHA','val'])

    df4.pivot("SHA", "CPU", "val").plot(kind='barh')

    #  PARA BENCHMARK 5
    valores50 = []
    muestra = open(outRoute+"/cpu0/b5/stats.txt",'r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores50.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.branchPred.condIncorrect"):
                valores50.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores50.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores50.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores50.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.num_reads::total"):
                valores50.append(math.log10(int(columna[1])))

    valores51 = []
    muestra = open(outRoute+"/cpu1/b5/stats.txt",'r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores51.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.branchPred.condIncorrect"):
                valores51.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores51.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores51.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores51.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.num_reads::total"):
                valores51.append(math.log10(int(columna[1])))
            
    valores52 =[]
    muestra = open(outRoute+"/cpu2/b5/stats.txt",'r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores52.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.branchPred.condIncorrect"):
                valores52.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores52.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores52.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores52.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.num_reads::total"):
                valores52.append(math.log10(int(columna[1])))
    
    valores53 =[]
    muestra = open(outRoute+"/cpu3/b5/stats.txt",'r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores53.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.branchPred.condIncorrect"):
                valores53.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores53.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores53.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores53.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.num_reads::total"):
                valores53.append(math.log10(int(columna[1])))

    valores54 =[]
    muestra = open(outRoute+"/cpu4/b5/stats.txt",'r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores54.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.branchPred.condIncorrect"):
                valores54.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores54.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores54.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores54.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.num_reads::total"):
                valores54.append(math.log10(int(columna[1])))
    
    valores55 =[]
    muestra = open(outRoute+"/cpu5/b5/stats.txt",'r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores55.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.branchPred.condIncorrect"):
                valores55.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores55.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores55.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores55.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.num_reads::total"):
                valores55.append(math.log10(int(columna[1])))
    

    df5 = pd.DataFrame([['CPU0','BranchMisses.',valores50[1]],['CPU0','CL1 d misses.',valores50[2]],
                       ['CPU0','CL1 i misses.',valores50[3]],['CPU0','CL2 misses',valores50[4]],['CPU0','RAM reads',valores50[5]],
                       ['CPU1','BranchMisses.',valores51[1]],['CPU1','CL1 d misses.',valores51[2]],
                       ['CPU1','CL1 i misses.',valores51[3]],['CPU1','CL2 misses',valores51[4]],['CPU1','RAM reads',valores51[5]],
                       ['CPU2','BranchMisses.',valores52[1]],['CPU2','CL1 d misses.',valores52[2]],
                       ['CPU2','CL1 i misses.',valores52[3]], ['CPU2','CL2 misses',valores52[4]],['CPU2','RAM reads',valores52[5]],
                       ['CPU3','BranchMisses.',valores53[1]],['CPU3','CL1 d misses.',valores53[2]],
                       ['CPU3','CL1 i misses.',valores53[3]],['CPU3','CL2 misses',valores53[4]],['CPU3','RAM reads',valores53[5]],
                       ['CPU4','BranchMisses.',valores54[1]],['CPU4','CL1 d misses.',valores54[2]],
                       ['CPU4','CL1 i misses.',valores54[3]],['CPU4','CL2 misses',valores54[4]],['CPU4','RAM reads',valores54[5]],
                       ['CPU5','BranchMisses.',valores55[1]],['CPU5','CL1 d misses.',valores55[2]],
                       ['CPU5','CL1 i misses.',valores55[3]],['CPU5','CL2 misses',valores55[4]],['CPU5','RAM reads',valores55[5]],
                       ], columns=['CPU','Queens','val'])

    df5.pivot("Queens", "CPU", "val").plot(kind='barh')    
    plt.show()
    
graph()


