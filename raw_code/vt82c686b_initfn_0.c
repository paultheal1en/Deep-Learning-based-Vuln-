static int vt82c686b_initfn(PCIDevice *d)

{

    uint8_t *pci_conf;

    uint8_t *wmask;

    int i;



    isa_bus_new(&d->qdev);



    pci_conf = d->config;

    pci_config_set_vendor_id(pci_conf, PCI_VENDOR_ID_VIA);

    pci_config_set_device_id(pci_conf, PCI_DEVICE_ID_VIA_ISA_BRIDGE);

    pci_config_set_class(pci_conf, PCI_CLASS_BRIDGE_ISA);

    pci_config_set_prog_interface(pci_conf, 0x0);

    pci_config_set_revision(pci_conf,0x40); /* Revision 4.0 */



    wmask = d->wmask;

    for (i = 0x00; i < 0xff; i++) {

       if (i<=0x03 || (i>=0x08 && i<=0x3f)) {

           wmask[i] = 0x00;

       }

    }



    qemu_register_reset(vt82c686b_reset, d);



    return 0;

}
