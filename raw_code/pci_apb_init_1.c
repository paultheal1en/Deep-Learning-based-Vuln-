PCIBus *pci_apb_init(target_phys_addr_t special_base,

                     target_phys_addr_t mem_base,

                     qemu_irq *pic, PCIBus **bus2, PCIBus **bus3)

{

    DeviceState *dev;

    SysBusDevice *s;

    APBState *d;



    /* Ultrasparc PBM main bus */

    dev = qdev_create(NULL, "pbm");

    qdev_init(dev);

    s = sysbus_from_qdev(dev);

    /* apb_config */

    sysbus_mmio_map(s, 0, special_base + 0x2000ULL);

    /* pci_ioport */

    sysbus_mmio_map(s, 1, special_base + 0x2000000ULL);

    /* mem_config: XXX size should be 4G-prom */

    sysbus_mmio_map(s, 2, special_base + 0x1000000ULL);

    /* mem_data */

    sysbus_mmio_map(s, 3, mem_base);

    d = FROM_SYSBUS(APBState, s);

    d->host_state.bus = pci_register_bus(&d->busdev.qdev, "pci",

                                         pci_apb_set_irq, pci_pbm_map_irq, pic,

                                         0, 32);

    pci_create_simple(d->host_state.bus, 0, "pbm");

    /* APB secondary busses */

    *bus2 = pci_bridge_init(d->host_state.bus, 8, PCI_VENDOR_ID_SUN,

                            PCI_DEVICE_ID_SUN_SIMBA, pci_apb_map_irq,

                            "Advanced PCI Bus secondary bridge 1");

    *bus3 = pci_bridge_init(d->host_state.bus, 9, PCI_VENDOR_ID_SUN,

                            PCI_DEVICE_ID_SUN_SIMBA, pci_apb_map_irq,

                            "Advanced PCI Bus secondary bridge 2");



    return d->host_state.bus;

}
