build_fadt(GArray *table_data, BIOSLinker *linker, AcpiPmInfo *pm,

           unsigned facs_tbl_offset, unsigned dsdt_tbl_offset,

           const char *oem_id, const char *oem_table_id)

{

    AcpiFadtDescriptorRev1 *fadt = acpi_data_push(table_data, sizeof(*fadt));

    unsigned fw_ctrl_offset = (char *)&fadt->firmware_ctrl - table_data->data;

    unsigned dsdt_entry_offset = (char *)&fadt->dsdt - table_data->data;



    /* FACS address to be filled by Guest linker */

    bios_linker_loader_add_pointer(linker,

        ACPI_BUILD_TABLE_FILE, fw_ctrl_offset, sizeof(fadt->firmware_ctrl),

        ACPI_BUILD_TABLE_FILE, facs_tbl_offset);



    /* DSDT address to be filled by Guest linker */

    fadt_setup(fadt, pm);

    bios_linker_loader_add_pointer(linker,

        ACPI_BUILD_TABLE_FILE, dsdt_entry_offset, sizeof(fadt->dsdt),

        ACPI_BUILD_TABLE_FILE, dsdt_tbl_offset);



    build_header(linker, table_data,

                 (void *)fadt, "FACP", sizeof(*fadt), 1, oem_id, oem_table_id);

}
