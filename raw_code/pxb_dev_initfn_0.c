static int pxb_dev_initfn(PCIDevice *dev)

{

    PXBDev *pxb = PXB_DEV(dev);

    DeviceState *ds, *bds;

    PCIBus *bus;

    const char *dev_name = NULL;



    if (pxb->numa_node != NUMA_NODE_UNASSIGNED &&

        pxb->numa_node >= nb_numa_nodes) {

        error_report("Illegal numa node %d.", pxb->numa_node);

        return -EINVAL;

    }



    if (dev->qdev.id && *dev->qdev.id) {

        dev_name = dev->qdev.id;

    }



    ds = qdev_create(NULL, TYPE_PXB_HOST);

    bus = pci_bus_new(ds, "pxb-internal", NULL, NULL, 0, TYPE_PXB_BUS);



    bus->parent_dev = dev;

    bus->address_space_mem = dev->bus->address_space_mem;

    bus->address_space_io = dev->bus->address_space_io;

    bus->map_irq = pxb_map_irq_fn;



    bds = qdev_create(BUS(bus), "pci-bridge");

    bds->id = dev_name;

    qdev_prop_set_uint8(bds, PCI_BRIDGE_DEV_PROP_CHASSIS_NR, pxb->bus_nr);

    qdev_prop_set_bit(bds, PCI_BRIDGE_DEV_PROP_SHPC, false);



    PCI_HOST_BRIDGE(ds)->bus = bus;



    if (pxb_register_bus(dev, bus)) {

        return -EINVAL;

    }



    qdev_init_nofail(ds);

    qdev_init_nofail(bds);



    pci_word_test_and_set_mask(dev->config + PCI_STATUS,

                               PCI_STATUS_66MHZ | PCI_STATUS_FAST_BACK);

    pci_config_set_class(dev->config, PCI_CLASS_BRIDGE_HOST);



    pxb_dev_list = g_list_insert_sorted(pxb_dev_list, pxb, pxb_compare);

    return 0;

}
