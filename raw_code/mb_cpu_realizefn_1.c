static void mb_cpu_realizefn(DeviceState *dev, Error **errp)

{

    MicroBlazeCPU *cpu = MICROBLAZE_CPU(dev);

    MicroBlazeCPUClass *mcc = MICROBLAZE_CPU_GET_CLASS(dev);



    cpu_reset(CPU(cpu));



    mcc->parent_realize(dev, errp);

}
