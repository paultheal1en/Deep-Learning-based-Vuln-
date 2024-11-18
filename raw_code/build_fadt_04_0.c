build_fadt(GArray *table_data, GArray *linker, unsigned dsdt)

{

    AcpiFadtDescriptorRev5_1 *fadt = acpi_data_push(table_data, sizeof(*fadt));



    /* Hardware Reduced = 1 and use PSCI 0.2+ and with HVC */

    fadt->flags = cpu_to_le32(1 << ACPI_FADT_F_HW_REDUCED_ACPI);

    fadt->arm_boot_flags = cpu_to_le16((1 << ACPI_FADT_ARM_USE_PSCI_G_0_2) |

                                       (1 << ACPI_FADT_ARM_PSCI_USE_HVC));



    /* ACPI v5.1 (fadt->revision.fadt->minor_revision) */

    fadt->minor_revision = 0x1;



    fadt->dsdt = cpu_to_le32(dsdt);

    /* DSDT address to be filled by Guest linker */

    bios_linker_loader_add_pointer(linker, ACPI_BUILD_TABLE_FILE,

                                   ACPI_BUILD_TABLE_FILE,

                                   table_data, &fadt->dsdt,

                                   sizeof fadt->dsdt);



    build_header(linker, table_data,

                 (void *)fadt, "FACP", sizeof(*fadt), 5, NULL, NULL);

}
