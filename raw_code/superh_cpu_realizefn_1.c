static void superh_cpu_realizefn(DeviceState *dev, Error **errp)

{

    SuperHCPU *cpu = SUPERH_CPU(dev);

    SuperHCPUClass *scc = SUPERH_CPU_GET_CLASS(dev);



    cpu_reset(CPU(cpu));



    scc->parent_realize(dev, errp);

}
