static void piix4_device_unplug_request_cb(HotplugHandler *hotplug_dev,

                                           DeviceState *dev, Error **errp)

{

    PIIX4PMState *s = PIIX4_PM(hotplug_dev);



    if (s->acpi_memory_hotplug.is_enabled &&

        object_dynamic_cast(OBJECT(dev), TYPE_PC_DIMM)) {

        acpi_memory_unplug_request_cb(hotplug_dev, &s->acpi_memory_hotplug,

                                      dev, errp);

    } else if (object_dynamic_cast(OBJECT(dev), TYPE_PCI_DEVICE)) {

        acpi_pcihp_device_unplug_cb(hotplug_dev, &s->acpi_pci_hotplug, dev,

                                    errp);

    } else if (object_dynamic_cast(OBJECT(dev), TYPE_CPU) &&

               !s->cpu_hotplug_legacy) {

        acpi_cpu_unplug_request_cb(hotplug_dev, &s->cpuhp_state, dev, errp);

    } else {

        error_setg(errp, "acpi: device unplug request for not supported device"

                   " type: %s", object_get_typename(OBJECT(dev)));

    }

}
