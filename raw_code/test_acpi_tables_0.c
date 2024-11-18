static void test_acpi_tables(test_data *data)

{

    int tables_nr = data->rsdt_tables_nr - 1; /* fadt is first */

    int i;



    for (i = 0; i < tables_nr; i++) {

        AcpiSdtTable ssdt_table;

        uint32_t addr;



        addr = le32_to_cpu(data->rsdt_tables_addr[i + 1]); /* fadt is first */

        test_dst_table(&ssdt_table, addr);

        g_array_append_val(data->tables, ssdt_table);

    }

}
