static void pxb_dev_realize(PCIDevice *dev, Error **errp)

{

    if (pci_bus_is_express(dev->bus)) {

        error_setg(errp, "pxb devices cannot reside on a PCIe bus");

        return;

    }



    pxb_dev_realize_common(dev, false, errp);

}
