static void piix4_ide_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);



    k->no_hotplug = 1;

    k->init = pci_piix_ide_initfn;

    k->exit = pci_piix_ide_exitfn;

    k->vendor_id = PCI_VENDOR_ID_INTEL;

    k->device_id = PCI_DEVICE_ID_INTEL_82371AB;

    k->class_id = PCI_CLASS_STORAGE_IDE;

    set_bit(DEVICE_CATEGORY_STORAGE, dc->categories);

    dc->no_user = 1;

}
