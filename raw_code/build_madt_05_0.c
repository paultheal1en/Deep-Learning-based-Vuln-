build_madt(GArray *table_data, BIOSLinker *linker, VirtGuestInfo *guest_info)

{

    int madt_start = table_data->len;

    const MemMapEntry *memmap = guest_info->memmap;

    const int *irqmap = guest_info->irqmap;

    AcpiMultipleApicTable *madt;

    AcpiMadtGenericDistributor *gicd;

    AcpiMadtGenericMsiFrame *gic_msi;

    int i;



    madt = acpi_data_push(table_data, sizeof *madt);



    gicd = acpi_data_push(table_data, sizeof *gicd);

    gicd->type = ACPI_APIC_GENERIC_DISTRIBUTOR;

    gicd->length = sizeof(*gicd);

    gicd->base_address = memmap[VIRT_GIC_DIST].base;

    gicd->version = guest_info->gic_version;



    for (i = 0; i < guest_info->smp_cpus; i++) {

        AcpiMadtGenericInterrupt *gicc = acpi_data_push(table_data,

                                                     sizeof *gicc);

        ARMCPU *armcpu = ARM_CPU(qemu_get_cpu(i));



        gicc->type = ACPI_APIC_GENERIC_INTERRUPT;

        gicc->length = sizeof(*gicc);

        if (guest_info->gic_version == 2) {

            gicc->base_address = memmap[VIRT_GIC_CPU].base;

        }

        gicc->cpu_interface_number = i;

        gicc->arm_mpidr = armcpu->mp_affinity;

        gicc->uid = i;

        gicc->flags = cpu_to_le32(ACPI_GICC_ENABLED);



        if (armcpu->has_pmu) {

            gicc->performance_interrupt = cpu_to_le32(PPI(VIRTUAL_PMU_IRQ));

        }

    }



    if (guest_info->gic_version == 3) {

        AcpiMadtGenericTranslator *gic_its;

        AcpiMadtGenericRedistributor *gicr = acpi_data_push(table_data,

                                                         sizeof *gicr);



        gicr->type = ACPI_APIC_GENERIC_REDISTRIBUTOR;

        gicr->length = sizeof(*gicr);

        gicr->base_address = cpu_to_le64(memmap[VIRT_GIC_REDIST].base);

        gicr->range_length = cpu_to_le32(memmap[VIRT_GIC_REDIST].size);



        if (its_class_name()) {

            gic_its = acpi_data_push(table_data, sizeof *gic_its);

            gic_its->type = ACPI_APIC_GENERIC_TRANSLATOR;

            gic_its->length = sizeof(*gic_its);

            gic_its->translation_id = 0;

            gic_its->base_address = cpu_to_le64(memmap[VIRT_GIC_ITS].base);

        }

    } else {

        gic_msi = acpi_data_push(table_data, sizeof *gic_msi);

        gic_msi->type = ACPI_APIC_GENERIC_MSI_FRAME;

        gic_msi->length = sizeof(*gic_msi);

        gic_msi->gic_msi_frame_id = 0;

        gic_msi->base_address = cpu_to_le64(memmap[VIRT_GIC_V2M].base);

        gic_msi->flags = cpu_to_le32(1);

        gic_msi->spi_count = cpu_to_le16(NUM_GICV2M_SPIS);

        gic_msi->spi_base = cpu_to_le16(irqmap[VIRT_GIC_V2M] + ARM_SPI_BASE);

    }



    build_header(linker, table_data,

                 (void *)(table_data->data + madt_start), "APIC",

                 table_data->len - madt_start, 3, NULL, NULL);

}
