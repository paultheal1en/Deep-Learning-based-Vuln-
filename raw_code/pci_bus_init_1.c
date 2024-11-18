static void pci_bus_init(PCIBus *bus, DeviceState *parent,

                         const char *name,

                         MemoryRegion *address_space_mem,

                         MemoryRegion *address_space_io,

                         uint8_t devfn_min)

{

    assert(PCI_FUNC(devfn_min) == 0);

    bus->devfn_min = devfn_min;

    bus->address_space_mem = address_space_mem;

    bus->address_space_io = address_space_io;





    memory_region_init_io(&bus->master_abort_mem, OBJECT(bus),

                          &master_abort_mem_ops, bus, "pci-master-abort",

                          memory_region_size(bus->address_space_mem));

    memory_region_add_subregion_overlap(bus->address_space_mem,

                                        0, &bus->master_abort_mem,

                                        MASTER_ABORT_MEM_PRIORITY);



    /* host bridge */

    QLIST_INIT(&bus->child);



    pci_host_bus_register(bus, parent);



    vmstate_register(NULL, -1, &vmstate_pcibus, bus);

}
