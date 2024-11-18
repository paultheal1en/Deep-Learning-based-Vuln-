static int piix4_initfn(PCIDevice *d)

{

    uint8_t *pci_conf;



    isa_bus_new(&d->qdev);

    register_savevm("PIIX4", 0, 2, piix_save, piix_load, d);



    pci_conf = d->config;

    pci_config_set_vendor_id(pci_conf, PCI_VENDOR_ID_INTEL);

    pci_config_set_device_id(pci_conf, PCI_DEVICE_ID_INTEL_82371AB_0); // 82371AB/EB/MB PIIX4 PCI-to-ISA bridge

    pci_config_set_class(pci_conf, PCI_CLASS_BRIDGE_ISA);

    pci_conf[PCI_HEADER_TYPE] =

        PCI_HEADER_TYPE_NORMAL | PCI_HEADER_TYPE_MULTI_FUNCTION; // header_type = PCI_multifunction, generic



    piix4_dev = d;

    piix4_reset(d);

    qemu_register_reset(piix4_reset, d);

    return 0;

}
