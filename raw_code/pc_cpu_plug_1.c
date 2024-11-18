static void pc_cpu_plug(HotplugHandler *hotplug_dev,

                        DeviceState *dev, Error **errp)

{

    CPUArchId *found_cpu;

    HotplugHandlerClass *hhc;

    Error *local_err = NULL;

    PCMachineState *pcms = PC_MACHINE(hotplug_dev);



    if (pcms->acpi_dev) {

        hhc = HOTPLUG_HANDLER_GET_CLASS(pcms->acpi_dev);

        hhc->plug(HOTPLUG_HANDLER(pcms->acpi_dev), dev, &local_err);

        if (local_err) {

            goto out;

        }

    }



    /* increment the number of CPUs */

    pcms->boot_cpus++;

    if (dev->hotplugged) {

        rtc_set_cpus_count(pcms->rtc, pcms->boot_cpus);

        fw_cfg_modify_i16(pcms->fw_cfg, FW_CFG_NB_CPUS, pcms->boot_cpus);

    }



    found_cpu = pc_find_cpu_slot(pcms, CPU(dev), NULL);

    found_cpu->cpu = CPU(dev);

out:

    error_propagate(errp, local_err);

}
