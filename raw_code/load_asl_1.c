static bool load_asl(GArray *sdts, AcpiSdtTable *sdt)

{

    AcpiSdtTable *temp;

    GError *error = NULL;

    GString *command_line = g_string_new(iasl);

    gint fd;

    gchar *out, *out_err;

    gboolean ret;

    int i;



    fd = g_file_open_tmp("asl-XXXXXX.dsl", &sdt->asl_file, &error);

    g_assert_no_error(error);

    close(fd);



    /* build command line */

    g_string_append_printf(command_line, " -p %s ", sdt->asl_file);

    if (compare_signature(sdt, "DSDT") ||

        compare_signature(sdt, "SSDT")) {

        for (i = 0; i < sdts->len; ++i) {

            temp = &g_array_index(sdts, AcpiSdtTable, i);

            if (compare_signature(temp, "DSDT") ||

                compare_signature(temp, "SSDT")) {

                g_string_append_printf(command_line, "-e %s ", temp->aml_file);

            }

        }

    }

    g_string_append_printf(command_line, "-d %s", sdt->aml_file);



    /* pass 'out' and 'out_err' in order to be redirected */

    ret = g_spawn_command_line_sync(command_line->str, &out, &out_err, NULL, &error);

    g_assert_no_error(error);



    if (ret) {

        ret = g_file_get_contents(sdt->asl_file, (gchar **)&sdt->asl,

                                  &sdt->asl_len, &error);

        g_assert(ret);

        g_assert_no_error(error);

        g_assert(sdt->asl_len);

    }



    g_free(out);

    g_free(out_err);

    g_string_free(command_line, true);



    return !ret;

}
