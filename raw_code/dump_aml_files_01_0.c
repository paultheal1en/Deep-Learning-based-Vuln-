static void dump_aml_files(test_data *data)

{

    AcpiSdtTable *sdt;

    GError *error = NULL;

    gint fd;

    ssize_t ret;

    int i;



    for (i = 0; i < data->ssdt_tables->len; ++i) {

        sdt = &g_array_index(data->ssdt_tables, AcpiSdtTable, i);

        g_assert(sdt->aml);



        fd = g_file_open_tmp("aml-XXXXXX", &sdt->aml_file, &error);

        g_assert_no_error(error);



        ret = qemu_write_full(fd, sdt, sizeof(AcpiTableHeader));

        g_assert(ret == sizeof(AcpiTableHeader));

        ret = qemu_write_full(fd, sdt->aml, sdt->aml_len);

        g_assert(ret == sdt->aml_len);



        close(fd);

    }

}
