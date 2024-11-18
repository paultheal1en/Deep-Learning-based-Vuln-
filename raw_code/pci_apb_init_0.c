PCIBus *pci_apb_init(target_phys_addr_t special_base,

                     target_phys_addr_t mem_base,

                     qemu_irq *ivec_irqs, PCIBus **bus2, PCIBus **bus3,

                     qemu_irq **pbm_irqs)

{

    DeviceState *dev;

    SysBusDevice *s;

    APBState *d;

    PCIDevice *pci_dev;

    PCIBridge *br;



    /* Ultrasparc PBM main bus */

    dev = qdev_create(NULL, "pbm");

    qdev_init_nofail(dev);

    s = sysbus_from_qdev(dev);

    /* apb_config */

    sysbus_mmio_map(s, 0, special_base);

    /* PCI configuration space */

    sysbus_mmio_map(s, 1, special_base + 0x1000000ULL);

    /* pci_ioport */

    sysbus_mmio_map(s, 2, special_base + 0x2000000ULL);

    d = FROM_SYSBUS(APBState, s);



    memory_region_init(&d->pci_mmio, "pci-mmio", 0x100000000ULL);

    memory_region_add_subregion(get_system_memory(), mem_base, &d->pci_mmio);



    d->bus = pci_register_bus(&d->busdev.qdev, "pci",

                              pci_apb_set_irq, pci_pbm_map_irq, d,

                              &d->pci_mmio,

                              get_system_io(),

                              0, 32);



    *pbm_irqs = d->pbm_irqs;

    d->ivec_irqs = ivec_irqs;



    pci_create_simple(d->bus, 0, "pbm-pci");



    /* APB secondary busses */

    pci_dev = pci_create_multifunction(d->bus, PCI_DEVFN(1, 0), true,

                                   "pbm-bridge");

    br = DO_UPCAST(PCIBridge, dev, pci_dev);

    pci_bridge_map_irq(br, "Advanced PCI Bus secondary bridge 1",

                       pci_apb_map_irq);

    qdev_init_nofail(&pci_dev->qdev);

    *bus2 = pci_bridge_get_sec_bus(br);



    pci_dev = pci_create_multifunction(d->bus, PCI_DEVFN(1, 1), true,

                                   "pbm-bridge");

    br = DO_UPCAST(PCIBridge, dev, pci_dev);

    pci_bridge_map_irq(br, "Advanced PCI Bus secondary bridge 2",

                       pci_apb_map_irq);

    qdev_init_nofail(&pci_dev->qdev);

    *bus3 = pci_bridge_get_sec_bus(br);



    return d->bus;

}
