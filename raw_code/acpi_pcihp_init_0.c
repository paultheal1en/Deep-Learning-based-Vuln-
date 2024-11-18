void acpi_pcihp_init(AcpiPciHpState *s, PCIBus *root_bus,

                     MemoryRegion *address_space_io)

{

    s->root= root_bus;

    memory_region_init_io(&s->io, NULL, &acpi_pcihp_io_ops, s,

                          "acpi-pci-hotplug",

                          PCI_HOTPLUG_SIZE);

    memory_region_add_subregion(address_space_io, PCI_HOTPLUG_ADDR, &s->io);

}
