static void build_fadt(GArray *table_data, BIOSLinker *linker,

                       VirtMachineState *vms, unsigned dsdt_tbl_offset)

{

    AcpiFadtDescriptorRev5_1 *fadt = acpi_data_push(table_data, sizeof(*fadt));

    unsigned dsdt_entry_offset = (char *)&fadt->dsdt - table_data->data;

    uint16_t bootflags;



    switch (vms->psci_conduit) {

    case QEMU_PSCI_CONDUIT_DISABLED:

        bootflags = 0;

        break;

    case QEMU_PSCI_CONDUIT_HVC:

        bootflags = ACPI_FADT_ARM_PSCI_COMPLIANT | ACPI_FADT_ARM_PSCI_USE_HVC;

        break;

    case QEMU_PSCI_CONDUIT_SMC:

        bootflags = ACPI_FADT_ARM_PSCI_COMPLIANT;

        break;

    default:

        g_assert_not_reached();

    }



    /* Hardware Reduced = 1 and use PSCI 0.2+ */

    fadt->flags = cpu_to_le32(1 << ACPI_FADT_F_HW_REDUCED_ACPI);

    fadt->arm_boot_flags = cpu_to_le16(bootflags);



    /* ACPI v5.1 (fadt->revision.fadt->minor_revision) */

    fadt->minor_revision = 0x1;



    /* DSDT address to be filled by Guest linker */

    bios_linker_loader_add_pointer(linker,

        ACPI_BUILD_TABLE_FILE, dsdt_entry_offset, sizeof(fadt->dsdt),

        ACPI_BUILD_TABLE_FILE, dsdt_tbl_offset);



    build_header(linker, table_data,

                 (void *)fadt, "FACP", sizeof(*fadt), 5, NULL, NULL);

}
