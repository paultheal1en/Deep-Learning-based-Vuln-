static void piix4_acpi_system_hot_add_init(MemoryRegion *parent,

                                           PCIBus *bus, PIIX4PMState *s)

{

    memory_region_init_io(&s->io_gpe, OBJECT(s), &piix4_gpe_ops, s,

                          "acpi-gpe0", GPE_LEN);

    memory_region_add_subregion(parent, GPE_BASE, &s->io_gpe);



    acpi_pcihp_init(OBJECT(s), &s->acpi_pci_hotplug, bus, parent,

                    s->use_acpi_pci_hotplug);



    s->cpu_hotplug_legacy = true;

    object_property_add_bool(OBJECT(s), "cpu-hotplug-legacy",

                             piix4_get_cpu_hotplug_legacy,

                             piix4_set_cpu_hotplug_legacy,

                             NULL);

    legacy_acpi_cpu_hotplug_init(parent, OBJECT(s), &s->gpe_cpu,

                                 PIIX4_CPU_HOTPLUG_IO_BASE);



    if (s->acpi_memory_hotplug.is_enabled) {

        acpi_memory_hotplug_init(parent, OBJECT(s), &s->acpi_memory_hotplug);

    }

}
