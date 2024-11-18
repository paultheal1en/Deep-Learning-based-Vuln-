static void piix4_pm_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);



    k->no_hotplug = 1;

    k->init = piix4_pm_initfn;

    k->config_write = pm_write_config;

    k->vendor_id = PCI_VENDOR_ID_INTEL;

    k->device_id = PCI_DEVICE_ID_INTEL_82371AB_3;

    k->revision = 0x03;

    k->class_id = PCI_CLASS_BRIDGE_OTHER;

    dc->desc = "PM";

    dc->no_user = 1;

    dc->vmsd = &vmstate_acpi;

    dc->props = piix4_pm_properties;

}
