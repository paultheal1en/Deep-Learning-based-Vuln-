static void s390_cpu_plug(HotplugHandler *hotplug_dev,
                        DeviceState *dev, Error **errp)
{
    MachineState *ms = MACHINE(hotplug_dev);
    S390CPU *cpu = S390_CPU(dev);
    g_assert(!ms->possible_cpus->cpus[cpu->env.core_id].cpu);
    ms->possible_cpus->cpus[cpu->env.core_id].cpu = OBJECT(dev);