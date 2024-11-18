void spapr_core_pre_plug(HotplugHandler *hotplug_dev, DeviceState *dev,

                         Error **errp)

{

    MachineState *machine = MACHINE(OBJECT(hotplug_dev));

    sPAPRMachineClass *smc = SPAPR_MACHINE_GET_CLASS(OBJECT(hotplug_dev));

    sPAPRMachineState *spapr = SPAPR_MACHINE(OBJECT(hotplug_dev));

    int spapr_max_cores = max_cpus / smp_threads;

    int index;

    Error *local_err = NULL;

    CPUCore *cc = CPU_CORE(dev);

    char *base_core_type = spapr_get_cpu_core_type(machine->cpu_model);

    const char *type = object_get_typename(OBJECT(dev));



    if (strcmp(base_core_type, type)) {

        error_setg(&local_err, "CPU core type should be %s", base_core_type);

        goto out;

    }



    if (!smc->dr_cpu_enabled && dev->hotplugged) {

        error_setg(&local_err, "CPU hotplug not supported for this machine");

        goto out;

    }



    if (cc->nr_threads != smp_threads) {

        error_setg(&local_err, "threads must be %d", smp_threads);

        goto out;

    }



    if (cc->core_id % smp_threads) {

        error_setg(&local_err, "invalid core id %d\n", cc->core_id);

        goto out;

    }



    index = cc->core_id / smp_threads;

    if (index < 0 || index >= spapr_max_cores) {

        error_setg(&local_err, "core id %d out of range", cc->core_id);

        goto out;

    }



    if (spapr->cores[index]) {

        error_setg(&local_err, "core %d already populated", cc->core_id);

        goto out;

    }



out:

    g_free(base_core_type);

    error_propagate(errp, local_err);

}
