int pci_bridge_initfn(PCIDevice *dev)

{

    PCIBus *parent = dev->bus;

    PCIBridge *br = DO_UPCAST(PCIBridge, dev, dev);

    PCIBus *sec_bus = &br->sec_bus;



    pci_set_word(dev->config + PCI_STATUS,

                 PCI_STATUS_66MHZ | PCI_STATUS_FAST_BACK);

    pci_config_set_class(dev->config, PCI_CLASS_BRIDGE_PCI);

    dev->config[PCI_HEADER_TYPE] =

        (dev->config[PCI_HEADER_TYPE] & PCI_HEADER_TYPE_MULTI_FUNCTION) |

        PCI_HEADER_TYPE_BRIDGE;

    pci_set_word(dev->config + PCI_SEC_STATUS,

                 PCI_STATUS_66MHZ | PCI_STATUS_FAST_BACK);



    qbus_create_inplace(&sec_bus->qbus, &pci_bus_info, &dev->qdev,

                        br->bus_name);

    sec_bus->parent_dev = dev;

    sec_bus->map_irq = br->map_irq;

    sec_bus->address_space_mem = &br->address_space_mem;

    memory_region_init(&br->address_space_mem, "pci_bridge_pci", INT64_MAX);

    sec_bus->address_space_io = &br->address_space_io;

    memory_region_init(&br->address_space_io, "pci_bridge_io", 65536);

    pci_bridge_region_init(br);

    QLIST_INIT(&sec_bus->child);

    QLIST_INSERT_HEAD(&parent->child, sec_bus, sibling);

    return 0;

}
