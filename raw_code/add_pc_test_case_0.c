static void add_pc_test_case(const char *mname)

{

    char *path;

    PlugTestData *data;



    if (!g_str_has_prefix(mname, "pc-")) {

        return;

    }

    data = g_new(PlugTestData, 1);

    data->machine = g_strdup(mname);

    data->cpu_model = "Haswell"; /* 1.3+ theoretically */

    data->sockets = 1;

    data->cores = 3;

    data->threads = 2;

    data->maxcpus = data->sockets * data->cores * data->threads * 2;

    if (g_str_has_suffix(mname, "-1.4") ||

        (strcmp(mname, "pc-1.3") == 0) ||

        (strcmp(mname, "pc-1.2") == 0) ||

        (strcmp(mname, "pc-1.1") == 0) ||

        (strcmp(mname, "pc-1.0") == 0) ||

        (strcmp(mname, "pc-0.15") == 0) ||

        (strcmp(mname, "pc-0.14") == 0) ||

        (strcmp(mname, "pc-0.13") == 0) ||

        (strcmp(mname, "pc-0.12") == 0) ||

        (strcmp(mname, "pc-0.11") == 0) ||

        (strcmp(mname, "pc-0.10") == 0)) {

        path = g_strdup_printf("cpu/%s/init/%ux%ux%u&maxcpus=%u",

                               mname, data->sockets, data->cores,

                               data->threads, data->maxcpus);

        qtest_add_data_func_full(path, data, test_plug_without_cpu_add,

                                 test_data_free);

        g_free(path);

    } else {

        path = g_strdup_printf("cpu/%s/add/%ux%ux%u&maxcpus=%u",

                               mname, data->sockets, data->cores,

                               data->threads, data->maxcpus);

        qtest_add_data_func_full(path, data, test_plug_with_cpu_add,

                                 test_data_free);

        g_free(path);

    }

}
