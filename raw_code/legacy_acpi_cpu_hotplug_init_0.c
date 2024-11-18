void legacy_acpi_cpu_hotplug_init(MemoryRegion *parent, Object *owner,

                                  AcpiCpuHotplug *gpe_cpu, uint16_t base)

{

    CPUState *cpu;



    CPU_FOREACH(cpu) {

        acpi_set_cpu_present_bit(gpe_cpu, cpu, &error_abort);

    }

    memory_region_init_io(&gpe_cpu->io, owner, &AcpiCpuHotplug_ops,

                          gpe_cpu, "acpi-cpu-hotplug", ACPI_GPE_PROC_LEN);

    memory_region_add_subregion(parent, base, &gpe_cpu->io);

    gpe_cpu->device = owner;

}
