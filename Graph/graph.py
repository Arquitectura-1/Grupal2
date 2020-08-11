import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
valores =[]
def graph():
    #  PARA BENCHMARK 1
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
    
    valores3 =[]
    muestra = open('/home/skynet/m5out/stats2.txt','r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores3.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.BranchMispred"):
                valores3.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores3.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores3.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores3.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.bytes_inst_read::total"):
                valores3.append(math.log10(int(columna[1])))

    valores4 =[]
    muestra = open('/home/skynet/m5out/stats2.txt','r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores4.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.BranchMispred"):
                valores4.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores4.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores4.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores4.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.bytes_inst_read::total"):
                valores4.append(math.log10(int(columna[1])))
    
    valores5 =[]
    muestra = open('/home/skynet/m5out/stats2.txt','r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores5.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.BranchMispred"):
                valores5.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores5.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores5.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores5.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.bytes_inst_read::total"):
                valores5.append(math.log10(int(columna[1])))
    
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(100,100))
    plt.subplots_adjust(left=0.03, bottom=0.327, right=0.977, hspace=0.88)

    df = pd.DataFrame([['CPU1','Tiempo Ejecución(Ticks)',valores[0]],['CPU1','CL1 d misses',valores[1]],['CPU1','CL1 i misses',valores[2]],
                       ['CPU1','CL2 misses',valores[3]],['CPU1','Accesos a RAM',valores[4]],['CPU1','BranchMisses',valores[5]],
                       ['CPU2','Tiempo Ejecución(Ticks)',valores2[0]], ['CPU2','CL1 d misses',valores2[1]],['CPU2','CL1 i misses',valores2[2]],
                       ['CPU2','CL2 misses',valores2[3]], ['CPU2','Accesos a RAM',valores2[4]],['CPU2','BranchMisses',valores2[5]],
                       ['CPU3','Tiempo Ejecución(Ticks)',valores3[0]],['CPU3','CL1 d misses',valores3[1]],['CPU3','CL1 i misses',valores3[2]],
                       ['CPU3','CL2 misses',valores3[3]],['CPU3','Accesos a RAM',valores3[4]],['CPU3','BranchMisses',valores3[5]],
                       ['CPU4','Tiempo Ejecución(Ticks)',valores4[0]],['CPU4','CL1 d misses',valores4[1]],['CPU4','CL1 i misses',valores4[2]],
                       ['CPU4','CL2 misses',valores4[3]],['CPU4','Accesos a RAM',valores4[4]],['CPU4','BranchMisses',valores4[5]],
                       ['CPU5','Tiempo Ejecución(Ticks)',valores5[0]],['CPU5','CL1 d misses',valores5[1]],['CPU5','CL1 i misses',valores5[2]],
                       ['CPU5','CL2 misses',valores5[3]],['CPU5','Accesos a RAM',valores5[4]],['CPU5','BranchMisses',valores5[5]],
                       ], columns=['CPU','VALOR(log10)','val'])

    df.pivot("VALOR(log10)", "CPU", "val").plot(kind='bar', ax=axes[0])

    #  PARA BENCHMARK 2
    valores21 = []
    muestra = open('/home/skynet/m5out/stats.txt','r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores21.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.BranchMispred"):
                valores21.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores21.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores21.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores21.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.bytes_inst_read::total"):
                valores21.append(math.log10(int(columna[1])))
            
    valores22 =[]
    muestra = open('/home/skynet/m5out/stats2.txt','r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores22.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.BranchMispred"):
                valores22.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores22.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores22.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores22.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.bytes_inst_read::total"):
                valores22.append(math.log10(int(columna[1])))
    
    valores23 =[]
    muestra = open('/home/skynet/m5out/stats2.txt','r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores23.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.BranchMispred"):
                valores23.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores23.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores23.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores23.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.bytes_inst_read::total"):
                valores23.append(math.log10(int(columna[1])))

    valores24 =[]
    muestra = open('/home/skynet/m5out/stats2.txt','r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores24.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.BranchMispred"):
                valores24.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores24.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores24.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores24.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.bytes_inst_read::total"):
                valores24.append(math.log10(int(columna[1])))
    
    valores25 =[]
    muestra = open('/home/skynet/m5out/stats2.txt','r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores25.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.BranchMispred"):
                valores25.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores25.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores25.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores25.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.bytes_inst_read::total"):
                valores25.append(math.log10(int(columna[1])))
    

    df2 = pd.DataFrame([['CPU1','Tiempo Ejecución(Ticks)',valores21[0]],['CPU1','CL1 d misses',valores21[1]],['CPU1','CL1 i misses',valores21[2]],
                       ['CPU1','CL2 misses',valores21[3]],['CPU1','Accesos a RAM',valores21[4]],['CPU1','BranchMisses',valores21[5]],
                       ['CPU2','Tiempo Ejecución(Ticks)',valores22[0]], ['CPU2','CL1 d misses',valores22[1]],['CPU2','CL1 i misses',valores22[2]],
                       ['CPU2','CL2 misses',valores22[3]], ['CPU2','Accesos a RAM',valores22[4]],['CPU2','BranchMisses',valores22[5]],
                       ['CPU3','Tiempo Ejecución(Ticks)',valores23[0]],['CPU3','CL1 d misses',valores23[1]],['CPU3','CL1 i misses',valores23[2]],
                       ['CPU3','CL2 misses',valores23[3]],['CPU3','Accesos a RAM',valores23[4]],['CPU3','BranchMisses',valores23[5]],
                       ['CPU4','Tiempo Ejecución(Ticks)',valores24[0]],['CPU4','CL1 d misses',valores24[1]],['CPU4','CL1 i misses',valores24[2]],
                       ['CPU4','CL2 misses',valores24[3]],['CPU4','Accesos a RAM',valores24[4]],['CPU4','BranchMisses',valores24[5]],
                       ['CPU5','Tiempo Ejecución(Ticks)',valores25[0]],['CPU5','CL1 d misses',valores25[1]],['CPU5','CL1 i misses',valores25[2]],
                       ['CPU5','CL2 misses',valores25[3]],['CPU5','Accesos a RAM',valores25[4]],['CPU5','BranchMisses',valores25[5]],
                       ], columns=['CPU','VALOR(log10)','val'])

    df2.pivot("VALOR(log10)", "CPU", "val").plot(kind='bar', ax=axes[1])

    #  PARA BENCHMARK 3
    valores31 = []
    muestra = open('/home/skynet/m5out/stats.txt','r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores31.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.BranchMispred"):
                valores31.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores31.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores31.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores31.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.bytes_inst_read::total"):
                valores31.append(math.log10(int(columna[1])))
            
    valores32 =[]
    muestra = open('/home/skynet/m5out/stats2.txt','r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores32.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.BranchMispred"):
                valores32.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores32.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores32.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores32.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.bytes_inst_read::total"):
                valores32.append(math.log10(int(columna[1])))
    
    valores33 =[]
    muestra = open('/home/skynet/m5out/stats2.txt','r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores33.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.BranchMispred"):
                valores33.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores33.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores33.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores33.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.bytes_inst_read::total"):
                valores33.append(math.log10(int(columna[1])))

    valores34 =[]
    muestra = open('/home/skynet/m5out/stats2.txt','r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores34.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.BranchMispred"):
                valores34.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores34.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores34.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores34.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.bytes_inst_read::total"):
                valores34.append(math.log10(int(columna[1])))
    
    valores35 =[]
    muestra = open('/home/skynet/m5out/stats2.txt','r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores35.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.BranchMispred"):
                valores35.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores35.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores35.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores35.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.bytes_inst_read::total"):
                valores35.append(math.log10(int(columna[1])))
    

    df3 = pd.DataFrame([['CPU1','Tiempo Ejecución(Ticks)',valores31[0]],['CPU1','CL1 d misses',valores31[1]],['CPU1','CL1 i misses',valores31[2]],
                       ['CPU1','CL2 misses',valores31[3]],['CPU1','Accesos a RAM',valores31[4]],['CPU1','BranchMisses',valores31[5]],
                       ['CPU2','Tiempo Ejecución(Ticks)',valores32[0]], ['CPU2','CL1 d misses',valores32[1]],['CPU2','CL1 i misses',valores32[2]],
                       ['CPU2','CL2 misses',valores32[3]], ['CPU2','Accesos a RAM',valores32[4]],['CPU2','BranchMisses',valores32[5]],
                       ['CPU3','Tiempo Ejecución(Ticks)',valores33[0]],['CPU3','CL1 d misses',valores33[1]],['CPU3','CL1 i misses',valores33[2]],
                       ['CPU3','CL2 misses',valores33[3]],['CPU3','Accesos a RAM',valores33[4]],['CPU3','BranchMisses',valores33[5]],
                       ['CPU4','Tiempo Ejecución(Ticks)',valores34[0]],['CPU4','CL1 d misses',valores34[1]],['CPU4','CL1 i misses',valores34[2]],
                       ['CPU4','CL2 misses',valores34[3]],['CPU4','Accesos a RAM',valores34[4]],['CPU4','BranchMisses',valores34[5]],
                       ['CPU5','Tiempo Ejecución(Ticks)',valores35[0]],['CPU5','CL1 d misses',valores35[1]],['CPU5','CL1 i misses',valores35[2]],
                       ['CPU5','CL2 misses',valores35[3]],['CPU5','Accesos a RAM',valores35[4]],['CPU5','BranchMisses',valores35[5]],
                       ], columns=['CPU','VALOR(log10)','val'])

    df3.pivot("VALOR(log10)", "CPU", "val").plot(kind='bar', ax=axes[2])


    #  PARA BENCHMARK 4
    valores41 = []
    muestra = open('/home/skynet/m5out/stats.txt','r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores41.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.BranchMispred"):
                valores41.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores41.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores41.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores41.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.bytes_inst_read::total"):
                valores41.append(math.log10(int(columna[1])))
            
    valores42 =[]
    muestra = open('/home/skynet/m5out/stats2.txt','r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores42.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.BranchMispred"):
                valores42.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores42.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores42.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores42.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.bytes_inst_read::total"):
                valores42.append(math.log10(int(columna[1])))
    
    valores43 =[]
    muestra = open('/home/skynet/m5out/stats2.txt','r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores43.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.BranchMispred"):
                valores43.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores43.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores43.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores43.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.bytes_inst_read::total"):
                valores43.append(math.log10(int(columna[1])))

    valores44 =[]
    muestra = open('/home/skynet/m5out/stats2.txt','r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores44.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.BranchMispred"):
                valores44.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores44.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores44.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores44.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.bytes_inst_read::total"):
                valores44.append(math.log10(int(columna[1])))
    
    valores45 =[]
    muestra = open('/home/skynet/m5out/stats2.txt','r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores45.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.BranchMispred"):
                valores45.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores45.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores45.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores45.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.bytes_inst_read::total"):
                valores45.append(math.log10(int(columna[1])))
    

    df4 = pd.DataFrame([['CPU1','Tiempo Ejecución(Ticks)',valores41[0]],['CPU1','CL1 d misses',valores41[1]],['CPU1','CL1 i misses',valores41[2]],
                       ['CPU1','CL2 misses',valores41[3]],['CPU1','Accesos a RAM',valores41[4]],['CPU1','BranchMisses',valores41[5]],
                       ['CPU2','Tiempo Ejecución(Ticks)',valores42[0]], ['CPU2','CL1 d misses',valores42[1]],['CPU2','CL1 i misses',valores42[2]],
                       ['CPU2','CL2 misses',valores42[3]], ['CPU2','Accesos a RAM',valores42[4]],['CPU2','BranchMisses',valores42[5]],
                       ['CPU3','Tiempo Ejecución(Ticks)',valores43[0]],['CPU3','CL1 d misses',valores43[1]],['CPU3','CL1 i misses',valores43[2]],
                       ['CPU3','CL2 misses',valores43[3]],['CPU3','Accesos a RAM',valores43[4]],['CPU3','BranchMisses',valores43[5]],
                       ['CPU4','Tiempo Ejecución(Ticks)',valores44[0]],['CPU4','CL1 d misses',valores44[1]],['CPU4','CL1 i misses',valores44[2]],
                       ['CPU4','CL2 misses',valores44[3]],['CPU4','Accesos a RAM',valores44[4]],['CPU4','BranchMisses',valores44[5]],
                       ['CPU5','Tiempo Ejecución(Ticks)',valores45[0]],['CPU5','CL1 d misses',valores45[1]],['CPU5','CL1 i misses',valores45[2]],
                       ['CPU5','CL2 misses',valores45[3]],['CPU5','Accesos a RAM',valores45[4]],['CPU5','BranchMisses',valores45[5]],
                       ], columns=['CPU','VALOR(log10)','val'])

    df4.pivot("VALOR(log10)", "CPU", "val").plot(kind='bar', ax=axes[3])

    #  PARA BENCHMARK 5
    valores51 = []
    muestra = open('/home/skynet/m5out/stats.txt','r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores51.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.BranchMispred"):
                valores51.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores51.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores51.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores51.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.bytes_inst_read::total"):
                valores51.append(math.log10(int(columna[1])))
            
    valores52 =[]
    muestra = open('/home/skynet/m5out/stats2.txt','r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores52.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.BranchMispred"):
                valores52.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores52.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores52.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores52.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.bytes_inst_read::total"):
                valores52.append(math.log10(int(columna[1])))
    
    valores53 =[]
    muestra = open('/home/skynet/m5out/stats2.txt','r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores53.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.BranchMispred"):
                valores53.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores53.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores53.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores53.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.bytes_inst_read::total"):
                valores53.append(math.log10(int(columna[1])))

    valores54 =[]
    muestra = open('/home/skynet/m5out/stats2.txt','r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores54.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.BranchMispred"):
                valores54.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores54.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores54.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores54.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.bytes_inst_read::total"):
                valores54.append(math.log10(int(columna[1])))
    
    valores55 =[]
    muestra = open('/home/skynet/m5out/stats2.txt','r')
    for fila in muestra:
        columna = fila.split()
        if (len(columna) != 0):
            if (columna[0] == "final_tick"):
                valores55.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.BranchMispred"):
                valores55.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.dcache.overall_misses::total"):
                valores55.append(math.log10(int(columna[1])))
            if (columna[0] == "system.cpu.icache.overall_misses::total"):
                valores55.append(math.log10(int(columna[1])))
            if (columna[0] == "system.l2.overall_misses::total"):
                valores55.append(math.log10(float(columna[1])))    
            if (columna[0] == "system.mem_ctrls.bytes_inst_read::total"):
                valores55.append(math.log10(int(columna[1])))
    

    df5 = pd.DataFrame([['CPU1','Tiempo Ejecución(Ticks)',valores51[0]],['CPU1','CL1 d misses',valores51[1]],['CPU1','CL1 i misses',valores51[2]],
                       ['CPU1','CL2 misses',valores51[3]],['CPU1','Accesos a RAM',valores51[4]],['CPU1','BranchMisses',valores51[5]],
                       ['CPU2','Tiempo Ejecución(Ticks)',valores52[0]], ['CPU2','CL1 d misses',valores52[1]],['CPU2','CL1 i misses',valores52[2]],
                       ['CPU2','CL2 misses',valores52[3]], ['CPU2','Accesos a RAM',valores52[4]],['CPU2','BranchMisses',valores52[5]],
                       ['CPU3','Tiempo Ejecución(Ticks)',valores53[0]],['CPU3','CL1 d misses',valores53[1]],['CPU3','CL1 i misses',valores53[2]],
                       ['CPU3','CL2 misses',valores53[3]],['CPU3','Accesos a RAM',valores53[4]],['CPU3','BranchMisses',valores53[5]],
                       ['CPU4','Tiempo Ejecución(Ticks)',valores54[0]],['CPU4','CL1 d misses',valores54[1]],['CPU4','CL1 i misses',valores54[2]],
                       ['CPU4','CL2 misses',valores54[3]],['CPU4','Accesos a RAM',valores54[4]],['CPU4','BranchMisses',valores54[5]],
                       ['CPU5','Tiempo Ejecución(Ticks)',valores55[0]],['CPU5','CL1 d misses',valores55[1]],['CPU5','CL1 i misses',valores55[2]],
                       ['CPU5','CL2 misses',valores55[3]],['CPU5','Accesos a RAM',valores55[4]],['CPU5','BranchMisses',valores55[5]],
                       ], columns=['CPU','VALOR(log10)','val'])

    df5.pivot("VALOR(log10)", "CPU", "val").plot(kind='bar', ax=axes[4])   
    plt.show()
 
graph()



