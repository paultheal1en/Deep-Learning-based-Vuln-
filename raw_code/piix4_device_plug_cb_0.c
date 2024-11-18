static void piix4_device_plug_cb(HotplugHandler *hotplug_dev,

                                 DeviceState *dev, Error **errp)

{

    PIIX4PMState *s = PIIX4_PM(hotplug_dev);



    if (s->acpi_memory_hotplug.is_enabled &&

        object_dynamic_cast(OBJECT(dev), TYPE_PC_DIMM)) {

        if (object_dynamic_cast(OBJECT(dev), TYPE_NVDIMM)) {

            nvdimm_acpi_plug_cb(hotplug_dev, dev);

        } else {

            acpi_memory_plug_cb(hotplug_dev, &s->acpi_memory_hotplug,

                                dev, errp);

        }

    } else if (object_dynamic_cast(OBJECT(dev), TYPE_PCI_DEVICE)) {

        if (!xen_enabled()) {

            acpi_pcihp_device_plug_cb(hotplug_dev, &s->acpi_pci_hotplug, dev,

                                      errp);

        }

    } else if (object_dynamic_cast(OBJECT(dev), TYPE_CPU)) {

        if (s->cpu_hotplug_legacy) {

            legacy_acpi_cpu_plug_cb(hotplug_dev, &s->gpe_cpu, dev, errp);

        } else {

            acpi_cpu_plug_cb(hotplug_dev, &s->cpuhp_state, dev, errp);

        }

    } else {

        error_setg(errp, "acpi: device plug request for not supported device"

                   " type: %s", object_get_typename(OBJECT(dev)));

    }

}
