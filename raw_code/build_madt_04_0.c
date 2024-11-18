build_madt(GArray *table_data, GArray *linker, VirtGuestInfo *guest_info)

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

    }



    if (guest_info->gic_version == 3) {

        AcpiMadtGenericRedistributor *gicr = acpi_data_push(table_data,

                                                         sizeof *gicr);



        gicr->type = ACPI_APIC_GENERIC_REDISTRIBUTOR;

        gicr->length = sizeof(*gicr);

        gicr->base_address = cpu_to_le64(memmap[VIRT_GIC_REDIST].base);

        gicr->range_length = cpu_to_le32(memmap[VIRT_GIC_REDIST].size);

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
