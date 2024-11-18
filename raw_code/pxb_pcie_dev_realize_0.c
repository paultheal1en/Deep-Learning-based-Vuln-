static void pxb_pcie_dev_realize(PCIDevice *dev, Error **errp)

{

    if (!pci_bus_is_express(dev->bus)) {

        error_setg(errp, "pxb-pcie devices cannot reside on a PCI bus");

        return;

    }



    pxb_dev_realize_common(dev, true, errp);

}
