static void alpha_cpu_realizefn(DeviceState *dev, Error **errp)
{
    AlphaCPUClass *acc = ALPHA_CPU_GET_CLASS(dev);
    acc->parent_realize(dev, errp);
}