static QObject *pci_get_dev_dict(PCIDevice *dev, PCIBus *bus, int bus_num)

{

    int class;

    QObject *obj;



    obj = qobject_from_jsonf("{ 'bus': %d, 'slot': %d, 'function': %d,"                                       "'class_info': %p, 'id': %p, 'regions': %p,"

                              " 'qdev_id': %s }",

                              bus_num,

                              PCI_SLOT(dev->devfn), PCI_FUNC(dev->devfn),

                              pci_get_dev_class(dev), pci_get_dev_id(dev),

                              pci_get_regions_list(dev),

                              dev->qdev.id ? dev->qdev.id : "");



    if (dev->config[PCI_INTERRUPT_PIN] != 0) {

        QDict *qdict = qobject_to_qdict(obj);

        qdict_put(qdict, "irq", qint_from_int(dev->config[PCI_INTERRUPT_LINE]));

    }



    class = pci_get_word(dev->config + PCI_CLASS_DEVICE);

    if (class == PCI_CLASS_BRIDGE_HOST || class == PCI_CLASS_BRIDGE_PCI) {

        QDict *qdict;

        QObject *pci_bridge;



        pci_bridge = qobject_from_jsonf("{ 'bus': "

        "{ 'number': %d, 'secondary': %d, 'subordinate': %d }, "

        "'io_range': { 'base': %" PRId64 ", 'limit': %" PRId64 "}, "

        "'memory_range': { 'base': %" PRId64 ", 'limit': %" PRId64 "}, "

        "'prefetchable_range': { 'base': %" PRId64 ", 'limit': %" PRId64 "} }",

        dev->config[PCI_PRIMARY_BUS], dev->config[PCI_SECONDARY_BUS],

        dev->config[PCI_SUBORDINATE_BUS],

        pci_bridge_get_base(dev, PCI_BASE_ADDRESS_SPACE_IO),

        pci_bridge_get_limit(dev, PCI_BASE_ADDRESS_SPACE_IO),

        pci_bridge_get_base(dev, PCI_BASE_ADDRESS_SPACE_MEMORY),

        pci_bridge_get_limit(dev, PCI_BASE_ADDRESS_SPACE_MEMORY),

        pci_bridge_get_base(dev, PCI_BASE_ADDRESS_SPACE_MEMORY |

                               PCI_BASE_ADDRESS_MEM_PREFETCH),

        pci_bridge_get_limit(dev, PCI_BASE_ADDRESS_SPACE_MEMORY |

                                PCI_BASE_ADDRESS_MEM_PREFETCH));



        if (dev->config[PCI_SECONDARY_BUS] != 0) {

            PCIBus *child_bus = pci_find_bus(bus, dev->config[PCI_SECONDARY_BUS]);



            if (child_bus) {

                qdict = qobject_to_qdict(pci_bridge);

                qdict_put_obj(qdict, "devices",

                              pci_get_devices_list(child_bus,

                                                   dev->config[PCI_SECONDARY_BUS]));

            }

        }

        qdict = qobject_to_qdict(obj);

        qdict_put_obj(qdict, "pci_bridge", pci_bridge);

    }



    return obj;

}
