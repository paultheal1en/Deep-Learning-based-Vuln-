static void ich_ahci_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);



    k->init = pci_ich9_ahci_init;

    k->exit = pci_ich9_uninit;

    k->config_write = pci_ich9_write_config;

    k->vendor_id = PCI_VENDOR_ID_INTEL;

    k->device_id = PCI_DEVICE_ID_INTEL_82801IR;

    k->revision = 0x02;

    k->class_id = PCI_CLASS_STORAGE_SATA;

    dc->alias = "ahci";

    dc->vmsd = &vmstate_ahci;

}
