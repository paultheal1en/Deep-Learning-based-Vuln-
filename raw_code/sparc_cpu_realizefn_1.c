static void sparc_cpu_realizefn(DeviceState *dev, Error **errp)
{
    SPARCCPUClass *scc = SPARC_CPU_GET_CLASS(dev);
    scc->parent_realize(dev, errp);
}