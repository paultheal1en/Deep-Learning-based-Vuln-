static void moxie_cpu_realizefn(DeviceState *dev, Error **errp)

{

    MoxieCPU *cpu = MOXIE_CPU(dev);

    MoxieCPUClass *mcc = MOXIE_CPU_GET_CLASS(dev);



    cpu_reset(CPU(cpu));



    mcc->parent_realize(dev, errp);

}
