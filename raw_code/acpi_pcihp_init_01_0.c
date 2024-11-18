void acpi_pcihp_init(AcpiPciHpState *s, PCIBus *root_bus,

                     MemoryRegion *address_space_io, bool bridges_enabled)

{

    uint16_t io_size = ACPI_PCIHP_SIZE;



    s->root= root_bus;

    s->legacy_piix = !bridges_enabled;



    if (s->legacy_piix) {

        unsigned *bus_bsel = g_malloc(sizeof *bus_bsel);



        io_size = ACPI_PCIHP_LEGACY_SIZE;



        *bus_bsel = ACPI_PCIHP_BSEL_DEFAULT;

        object_property_add_uint32_ptr(OBJECT(root_bus), ACPI_PCIHP_PROP_BSEL,

                                       bus_bsel, NULL);

    }



    memory_region_init_io(&s->io, NULL, &acpi_pcihp_io_ops, s,

                          "acpi-pci-hotplug", io_size);

    memory_region_add_subregion(address_space_io, ACPI_PCIHP_ADDR, &s->io);

}
