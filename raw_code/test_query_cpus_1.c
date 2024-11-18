static void test_query_cpus(const void *data)

{

    char *cli;

    QDict *resp;

    QList *cpus;

    const QObject *e;



    cli = make_cli(data, "-smp 8 -numa node,cpus=0-3 -numa node,cpus=4-7");

    qtest_start(cli);

    cpus = get_cpus(&resp);

    g_assert(cpus);



    while ((e = qlist_pop(cpus))) {

        QDict *cpu, *props;

        int64_t cpu_idx, node;



        cpu = qobject_to_qdict(e);

        g_assert(qdict_haskey(cpu, "CPU"));

        g_assert(qdict_haskey(cpu, "props"));



        cpu_idx = qdict_get_int(cpu, "CPU");

        props = qdict_get_qdict(cpu, "props");

        g_assert(qdict_haskey(props, "node-id"));

        node = qdict_get_int(props, "node-id");

        if (cpu_idx >= 0 && cpu_idx < 4) {

            g_assert_cmpint(node, ==, 0);

        } else {

            g_assert_cmpint(node, ==, 1);

        }

    }



    QDECREF(resp);

    qtest_end();

    g_free(cli);

}
