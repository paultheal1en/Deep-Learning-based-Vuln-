PCIBus *pci_grackle_init(uint32_t base, qemu_irq *pic,

                         MemoryRegion *address_space_mem,

                         MemoryRegion *address_space_io)

{

    DeviceState *dev;

    SysBusDevice *s;

    PCIHostState *phb;

    GrackleState *d;



    dev = qdev_create(NULL, TYPE_GRACKLE_PCI_HOST_BRIDGE);

    qdev_init_nofail(dev);

    s = SYS_BUS_DEVICE(dev);

    phb = PCI_HOST_BRIDGE(dev);

    d = GRACKLE_PCI_HOST_BRIDGE(dev);



    memory_region_init(&d->pci_mmio, OBJECT(s), "pci-mmio", 0x100000000ULL);

    memory_region_init_alias(&d->pci_hole, OBJECT(s), "pci-hole", &d->pci_mmio,

                             0x80000000ULL, 0x7e000000ULL);

    memory_region_add_subregion(address_space_mem, 0x80000000ULL,

                                &d->pci_hole);



    phb->bus = pci_register_bus(dev, "pci",

                                pci_grackle_set_irq,

                                pci_grackle_map_irq,

                                pic,

                                &d->pci_mmio,

                                address_space_io,

                                0, 4, TYPE_PCI_BUS);



    pci_create_simple(phb->bus, 0, "grackle");



    sysbus_mmio_map(s, 0, base);

    sysbus_mmio_map(s, 1, base + 0x00200000);



    return phb->bus;

}
