static int dec_21154_initfn(PCIDevice *dev)

{

    int rc;



    rc = pci_bridge_initfn(dev);

    if (rc < 0) {

        return rc;

    }



    pci_config_set_vendor_id(dev->config, PCI_VENDOR_ID_DEC);

    pci_config_set_device_id(dev->config, PCI_DEVICE_ID_DEC_21154);

    return 0;

}
