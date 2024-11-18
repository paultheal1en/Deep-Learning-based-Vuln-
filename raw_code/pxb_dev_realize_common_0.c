static void pxb_dev_realize_common(PCIDevice *dev, bool pcie, Error **errp)

{

    PXBDev *pxb = convert_to_pxb(dev);

    DeviceState *ds, *bds = NULL;

    PCIBus *bus;

    const char *dev_name = NULL;

    Error *local_err = NULL;



    if (pxb->numa_node != NUMA_NODE_UNASSIGNED &&

        pxb->numa_node >= nb_numa_nodes) {

        error_setg(errp, "Illegal numa node %d", pxb->numa_node);

        return;

    }



    if (dev->qdev.id && *dev->qdev.id) {

        dev_name = dev->qdev.id;

    }



    ds = qdev_create(NULL, TYPE_PXB_HOST);

    if (pcie) {

        bus = pci_root_bus_new(ds, dev_name, NULL, NULL, 0, TYPE_PXB_PCIE_BUS);

    } else {

        bus = pci_root_bus_new(ds, "pxb-internal", NULL, NULL, 0, TYPE_PXB_BUS);

        bds = qdev_create(BUS(bus), "pci-bridge");

        bds->id = dev_name;

        qdev_prop_set_uint8(bds, PCI_BRIDGE_DEV_PROP_CHASSIS_NR, pxb->bus_nr);

        qdev_prop_set_bit(bds, PCI_BRIDGE_DEV_PROP_SHPC, false);

    }



    bus->parent_dev = dev;

    bus->address_space_mem = dev->bus->address_space_mem;

    bus->address_space_io = dev->bus->address_space_io;

    bus->map_irq = pxb_map_irq_fn;



    PCI_HOST_BRIDGE(ds)->bus = bus;



    pxb_register_bus(dev, bus, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        goto err_register_bus;

    }



    qdev_init_nofail(ds);

    if (bds) {

        qdev_init_nofail(bds);

    }



    pci_word_test_and_set_mask(dev->config + PCI_STATUS,

                               PCI_STATUS_66MHZ | PCI_STATUS_FAST_BACK);

    pci_config_set_class(dev->config, PCI_CLASS_BRIDGE_HOST);



    pxb_dev_list = g_list_insert_sorted(pxb_dev_list, pxb, pxb_compare);

    return;



err_register_bus:

    object_unref(OBJECT(bds));

    object_unparent(OBJECT(bus));

    object_unref(OBJECT(ds));

}
