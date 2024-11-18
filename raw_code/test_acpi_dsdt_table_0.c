static void test_acpi_dsdt_table(test_data *data)

{

    AcpiSdtTable dsdt_table;

    uint32_t addr = le32_to_cpu(data->fadt_table.dsdt);



    test_dst_table(&dsdt_table, addr);

    ACPI_ASSERT_CMP(dsdt_table.header.signature, "DSDT");



    /* Since DSDT isn't in RSDT, add DSDT to ASL test tables list manually */

    g_array_append_val(data->tables, dsdt_table);

}
