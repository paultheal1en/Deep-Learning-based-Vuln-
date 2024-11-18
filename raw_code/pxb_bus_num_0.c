static int pxb_bus_num(PCIBus *bus)

{

    PXBDev *pxb = PXB_DEV(bus->parent_dev);



    return pxb->bus_nr;

}
