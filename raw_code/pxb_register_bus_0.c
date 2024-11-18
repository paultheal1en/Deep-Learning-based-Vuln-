static void pxb_register_bus(PCIDevice *dev, PCIBus *pxb_bus, Error **errp)

{

    PCIBus *bus = dev->bus;

    int pxb_bus_num = pci_bus_num(pxb_bus);



    if (bus->parent_dev) {

        error_setg(errp, "PXB devices can be attached only to root bus");

        return;

    }



    QLIST_FOREACH(bus, &bus->child, sibling) {

        if (pci_bus_num(bus) == pxb_bus_num) {

            error_setg(errp, "Bus %d is already in use", pxb_bus_num);

            return;

        }

    }

    QLIST_INSERT_HEAD(&dev->bus->child, pxb_bus, sibling);

}
