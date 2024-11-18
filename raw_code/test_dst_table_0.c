static void test_dst_table(AcpiSdtTable *sdt_table, uint32_t addr)

{

    uint8_t checksum;



    memset(sdt_table, 0, sizeof(*sdt_table));

    ACPI_READ_TABLE_HEADER(&sdt_table->header, addr);



    sdt_table->aml_len = le32_to_cpu(sdt_table->header.length)

                         - sizeof(AcpiTableHeader);

    sdt_table->aml = g_malloc0(sdt_table->aml_len);

    ACPI_READ_ARRAY_PTR(sdt_table->aml, sdt_table->aml_len, addr);



    checksum = acpi_calc_checksum((uint8_t *)sdt_table,

                                  sizeof(AcpiTableHeader)) +

               acpi_calc_checksum((uint8_t *)sdt_table->aml,

                                  sdt_table->aml_len);

    g_assert(!checksum);

}
