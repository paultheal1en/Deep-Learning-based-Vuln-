void pci_qdev_register(PCIDeviceInfo *info)

{

    info->qdev.init = pci_qdev_init;

    info->qdev.bus_type = BUS_TYPE_PCI;

    qdev_register(&info->qdev);

}
