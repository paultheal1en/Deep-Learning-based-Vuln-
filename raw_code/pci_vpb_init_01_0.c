static void pci_vpb_init(Object *obj)

{

    PCIHostState *h = PCI_HOST_BRIDGE(obj);

    PCIVPBState *s = PCI_VPB(obj);



    memory_region_init(&s->pci_io_space, OBJECT(s), "pci_io", 1ULL << 32);

    memory_region_init(&s->pci_mem_space, OBJECT(s), "pci_mem", 1ULL << 32);



    pci_bus_new_inplace(&s->pci_bus, sizeof(s->pci_bus), DEVICE(obj), "pci",

                        &s->pci_mem_space, &s->pci_io_space,

                        PCI_DEVFN(11, 0), TYPE_PCI_BUS);

    h->bus = &s->pci_bus;



    object_initialize(&s->pci_dev, sizeof(s->pci_dev), TYPE_VERSATILE_PCI_HOST);

    qdev_set_parent_bus(DEVICE(&s->pci_dev), BUS(&s->pci_bus));



    /* Window sizes for VersatilePB; realview_pci's init will override */

    s->mem_win_size[0] = 0x0c000000;

    s->mem_win_size[1] = 0x10000000;

    s->mem_win_size[2] = 0x10000000;

}
