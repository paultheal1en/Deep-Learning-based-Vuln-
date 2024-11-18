PCIHostState *spapr_create_phb(sPAPREnvironment *spapr, int index,

                               const char *busname)

{

    DeviceState *dev;



    dev = qdev_create(NULL, TYPE_SPAPR_PCI_HOST_BRIDGE);

    qdev_prop_set_uint32(dev, "index", index);

    qdev_prop_set_string(dev, "busname", busname);

    qdev_init_nofail(dev);



    return PCI_HOST_BRIDGE(dev);

}
