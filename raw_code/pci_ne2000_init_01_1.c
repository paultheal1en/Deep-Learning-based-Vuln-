PCIDevice *pci_ne2000_init(PCIBus *bus, NICInfo *nd, int devfn)

{

    PCINE2000State *d;

    NE2000State *s;

    uint8_t *pci_conf;



    d = (PCINE2000State *)pci_register_device(bus,

                                              "NE2000", sizeof(PCINE2000State),

                                              devfn,

                                              NULL, NULL);

    pci_conf = d->dev.config;

    pci_config_set_vendor_id(pci_conf, PCI_VENDOR_ID_REALTEK);

    pci_config_set_device_id(pci_conf, PCI_DEVICE_ID_REALTEK_8029);

    pci_config_set_class(pci_conf, PCI_CLASS_NETWORK_ETHERNET);

    pci_conf[0x0e] = 0x00; // header_type

    pci_conf[0x3d] = 1; // interrupt pin 0



    pci_register_io_region(&d->dev, 0, 0x100,

                           PCI_ADDRESS_SPACE_IO, ne2000_map);

    s = &d->ne2000;

    s->irq = d->dev.irq[0];

    s->pci_dev = (PCIDevice *)d;

    memcpy(s->macaddr, nd->macaddr, 6);

    ne2000_reset(s);

    s->vc = qemu_new_vlan_client(nd->vlan, nd->model, nd->name,

                                 ne2000_receive, ne2000_can_receive, s);



    qemu_format_nic_info_str(s->vc, s->macaddr);



    register_savevm("ne2000", -1, 3, ne2000_save, ne2000_load, s);



    return (PCIDevice *)d;

}
