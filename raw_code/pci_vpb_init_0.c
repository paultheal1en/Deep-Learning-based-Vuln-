static int pci_vpb_init(SysBusDevice *dev)

{

    PCIVPBState *s = FROM_SYSBUS(PCIVPBState, dev);

    PCIBus *bus;

    int i;



    for (i = 0; i < 4; i++) {

        sysbus_init_irq(dev, &s->irq[i]);

    }

    bus = pci_register_bus(&dev->qdev, "pci",

                           pci_vpb_set_irq, pci_vpb_map_irq, s->irq,

                           get_system_memory(), get_system_io(),

                           PCI_DEVFN(11, 0), 4);



    /* ??? Register memory space.  */



    memory_region_init_io(&s->mem_config, &pci_vpb_config_ops, bus,

                          "pci-vpb-selfconfig", 0x1000000);

    memory_region_init_io(&s->mem_config2, &pci_vpb_config_ops, bus,

                          "pci-vpb-config", 0x1000000);

    if (s->realview) {

        isa_mmio_setup(&s->isa, 0x0100000);

    }



    sysbus_init_mmio_cb2(dev, pci_vpb_map, pci_vpb_unmap);



    pci_create_simple(bus, -1, "versatile_pci_host");

    return 0;

}
