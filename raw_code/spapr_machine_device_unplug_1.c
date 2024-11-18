static void spapr_machine_device_unplug(HotplugHandler *hotplug_dev,

                                      DeviceState *dev, Error **errp)

{

    sPAPRMachineClass *smc = SPAPR_MACHINE_GET_CLASS(qdev_get_machine());



    if (object_dynamic_cast(OBJECT(dev), TYPE_PC_DIMM)) {

        error_setg(errp, "Memory hot unplug not supported by sPAPR");

    } else if (object_dynamic_cast(OBJECT(dev), TYPE_SPAPR_CPU_CORE)) {

        if (!smc->dr_cpu_enabled) {

            error_setg(errp, "CPU hot unplug not supported on this machine");

            return;

        }

        spapr_core_unplug(hotplug_dev, dev, errp);

    }

}
