static int pci_set_default_subsystem_id(PCIDevice *pci_dev)

{

    uint16_t *id;



    id = (void*)(&pci_dev->config[PCI_SUBSYSTEM_VENDOR_ID]);

    id[0] = cpu_to_le16(pci_default_sub_vendor_id);

    id[1] = cpu_to_le16(pci_default_sub_device_id);

    return 0;

}
