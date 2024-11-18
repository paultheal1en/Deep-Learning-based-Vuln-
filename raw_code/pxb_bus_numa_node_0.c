static uint16_t pxb_bus_numa_node(PCIBus *bus)

{

    PXBDev *pxb = PXB_DEV(bus->parent_dev);



    return pxb->numa_node;

}
