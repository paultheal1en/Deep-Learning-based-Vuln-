static int sh_pci_init_device(SysBusDevice *dev)

{

    SHPCIState *s;

    int i;



    s = FROM_SYSBUS(SHPCIState, dev);

    for (i = 0; i < 4; i++) {

        sysbus_init_irq(dev, &s->irq[i]);

    }

    s->bus = pci_register_bus(&s->busdev.qdev, "pci",

                              sh_pci_set_irq, sh_pci_map_irq,

                              s->irq,

                              get_system_memory(),

                              get_system_io(),

                              PCI_DEVFN(0, 0), 4);

    memory_region_init_io(&s->memconfig_p4, &sh_pci_reg_ops, s,

                          "sh_pci", 0x224);

    memory_region_init_alias(&s->memconfig_a7, "sh_pci.2", &s->memconfig_a7,

                             0, 0x224);

    isa_mmio_setup(&s->isa, 0x40000);

    sysbus_init_mmio_cb2(dev, sh_pci_map, sh_pci_unmap);

    sysbus_init_mmio_region(dev, &s->memconfig_a7);

    sysbus_init_mmio_region(dev, &s->isa);

    s->dev = pci_create_simple(s->bus, PCI_DEVFN(0, 0), "sh_pci_host");

    return 0;

}
