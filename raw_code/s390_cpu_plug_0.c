static void s390_cpu_plug(HotplugHandler *hotplug_dev,

                        DeviceState *dev, Error **errp)

{

    gchar *name;

    S390CPU *cpu = S390_CPU(dev);

    CPUState *cs = CPU(dev);



    name = g_strdup_printf("cpu[%i]", cpu->env.cpu_num);

    object_property_set_link(OBJECT(hotplug_dev), OBJECT(cs), name,

                             errp);

    g_free(name);

}
