static int piix3_initfn(PCIDevice *dev)

{

    PIIX3State *d = DO_UPCAST(PIIX3State, dev, dev);

    uint8_t *pci_conf;



    isa_bus_new(&d->dev.qdev);



    pci_conf = d->dev.config;

    pci_config_set_vendor_id(pci_conf, PCI_VENDOR_ID_INTEL);

    pci_config_set_device_id(pci_conf, PCI_DEVICE_ID_INTEL_82371SB_0); // 82371SB PIIX3 PCI-to-ISA bridge (Step A1)

    pci_config_set_class(pci_conf, PCI_CLASS_BRIDGE_ISA);

    pci_conf[PCI_HEADER_TYPE] =

        PCI_HEADER_TYPE_NORMAL | PCI_HEADER_TYPE_MULTI_FUNCTION; // header_type = PCI_multifunction, generic



    qemu_register_reset(piix3_reset, d);

    return 0;

}
