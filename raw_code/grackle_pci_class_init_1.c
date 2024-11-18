static void grackle_pci_class_init(ObjectClass *klass, void *data)

{

    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);

    DeviceClass *dc = DEVICE_CLASS(klass);



    k->init      = grackle_pci_host_init;

    k->vendor_id = PCI_VENDOR_ID_MOTOROLA;

    k->device_id = PCI_DEVICE_ID_MOTOROLA_MPC106;

    k->revision  = 0x00;

    k->class_id  = PCI_CLASS_BRIDGE_HOST;

    dc->no_user = 1;

}
