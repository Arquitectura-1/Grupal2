import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
valores =[]
def graph():
    muestra = open('/home/skynet/m5out/stats.txt','r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.BranchMispred"):
                valores.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.bytes_inst_read::total"):
                valores.append(math.log10(int(columna[1])))
            
    valores2 =[]
    muestra = open('/home/skynet/m5out/stats2.txt','r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores2.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.BranchMispred"):
                valores2.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores2.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores2.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores2.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.bytes_inst_read::total"):
                valores2.append(math.log10(int(columna[1])))
    
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(100,100))
    plt.subplots_adjust(left=0.03, bottom=0.327, right=0.977, hspace=0.88)

    df = pd.DataFrame([['CPU1','Tiempo Ejecución(Ticks)',valores[0]],['CPU1','CL1 d misses',valores[1]],['CPU1','CL1 i misses',valores[2]],
                       ['CPU1','CL2 misses',valores[3]],['CPU1','Accesos a RAM',valores[4]],['CPU1','BranchMisses',valores[5]],
                       ['CPU2','Tiempo Ejecución(Ticks)',valores2[0]], ['CPU2','CL1 d misses',valores2[1]],['CPU2','CL1 i misses',valores2[2]],
                       ['CPU2','CL2 misses',valores2[3]], ['CPU2','Accesos a RAM',valores2[4]],['CPU2','BranchMisses',valores2[5]]],
      
                       columns=['CPU','VALOR(log10)','val'])

    df.pivot("VALOR(log10)", "CPU", "val").plot(kind='bar', ax=axes[0])

    df2 = pd.DataFrame([['CPU1','Tiempo Ejecución(Ticks)',valores[0]],['CPU1','CL1 d misses',valores[1]],['CPU1','CL1 i misses',valores[2]],
                       ['CPU1','CL2 misses',valores[3]],['CPU1','Accesos a RAM',valores[4]],['CPU1','BranchMisses',valores[5]],
                       ['CPU2','Tiempo Ejecución(Ticks)',valores2[0]], ['CPU2','CL1 d misses',valores2[1]],['CPU2','CL1 i misses',valores2[2]],
                       ['CPU2','CL2 misses',valores2[3]], ['CPU2','Accesos a RAM',valores2[4]],['CPU2','BranchMisses',valores2[5]]],
      
                       columns=['CPU','VALOR(log10)','val'])

    df2.pivot("VALOR(log10)", "CPU", "val").plot(kind='bar', ax=axes[1])

    df3 = pd.DataFrame([['CPU1','Tiempo Ejecución(Ticks)',valores[0]],['CPU1','CL1 d misses',valores[1]],['CPU1','CL1 i misses',valores[2]],
                       ['CPU1','CL2 misses',valores[3]],['CPU1','Accesos a RAM',valores[4]],['CPU1','BranchMisses',valores[5]],
                       ['CPU2','Tiempo Ejecución(Ticks)',valores2[0]], ['CPU2','CL1 d misses',valores2[1]],['CPU2','CL1 i misses',valores2[2]],
                       ['CPU2','CL2 misses',valores2[3]], ['CPU2','Accesos a RAM',valores2[4]],['CPU2','BranchMisses',valores2[5]]],
      
                       columns=['CPU','VALOR(log10)','val'])

    df3.pivot("VALOR(log10)", "CPU", "val").plot(kind='bar', ax=axes[2])

    df4 = pd.DataFrame([['CPU1','Tiempo Ejecución(Ticks)',valores[0]],['CPU1','CL1 d misses',valores[1]],['CPU1','CL1 i misses',valores[2]],
                       ['CPU1','CL2 misses',valores[3]],['CPU1','Accesos a RAM',valores[4]],['CPU1','BranchMisses',valores[5]],
                       ['CPU2','Tiempo Ejecución(Ticks)',valores2[0]], ['CPU2','CL1 d misses',valores2[1]],['CPU2','CL1 i misses',valores2[2]],
                       ['CPU2','CL2 misses',valores2[3]], ['CPU2','Accesos a RAM',valores2[4]],['CPU2','BranchMisses',valores2[5]]],
      
                       columns=['CPU','VALOR(log10)','val'])

    df4.pivot("VALOR(log10)", "CPU", "val").plot(kind='bar', ax=axes[3])

    df5 = pd.DataFrame([['CPU1','Tiempo Ejecución(Ticks)',valores[0]],['CPU1','CL1 d misses',valores[1]],['CPU1','CL1 i misses',valores[2]],
                       ['CPU1','CL2 misses',valores[3]],['CPU1','Accesos a RAM',valores[4]],['CPU1','BranchMisses',valores[5]],
                       ['CPU2','Tiempo Ejecución(Ticks)',valores2[0]], ['CPU2','CL1 d misses',valores2[1]],['CPU2','CL1 i misses',valores2[2]],
                       ['CPU2','CL2 misses',valores2[3]], ['CPU2','Accesos a RAM',valores2[4]],['CPU2','BranchMisses',valores2[5]]],
      
                       columns=['CPU','Ben','val'])

    df5.pivot("Ben", "CPU", "val").plot(kind='bar', ax=axes[4])

    plt.show()
 

graph()



