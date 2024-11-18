static int usb_uhci_vt82c686b_initfn(PCIDevice *dev)

{

    UHCIState *s = DO_UPCAST(UHCIState, dev, dev);

    uint8_t *pci_conf = s->dev.config;



    pci_config_set_vendor_id(pci_conf, PCI_VENDOR_ID_VIA);

    pci_config_set_device_id(pci_conf, PCI_DEVICE_ID_VIA_UHCI);



    /* USB misc control 1/2 */

    pci_set_long(pci_conf + 0x40,0x00001000);

    /* PM capability */

    pci_set_long(pci_conf + 0x80,0x00020001);

    /* USB legacy support  */

    pci_set_long(pci_conf + 0xc0,0x00002000);



    return usb_uhci_common_initfn(s);

}
