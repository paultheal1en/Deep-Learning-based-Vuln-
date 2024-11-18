static void aarch64_numa_cpu(const void *data)

{

    char *cli;

    QDict *resp;

    QList *cpus;

    const QObject *e;



    cli = make_cli(data, "-smp 2 "

        "-numa node,nodeid=0 -numa node,nodeid=1 "

        "-numa cpu,node-id=1,thread-id=0 "

        "-numa cpu,node-id=0,thread-id=1");

    qtest_start(cli);

    cpus = get_cpus(&resp);

    g_assert(cpus);



    while ((e = qlist_pop(cpus))) {

        QDict *cpu, *props;

        int64_t thread, node;



        cpu = qobject_to_qdict(e);

        g_assert(qdict_haskey(cpu, "props"));

        props = qdict_get_qdict(cpu, "props");



        g_assert(qdict_haskey(props, "node-id"));

        node = qdict_get_int(props, "node-id");

        g_assert(qdict_haskey(props, "thread-id"));

        thread = qdict_get_int(props, "thread-id");



        if (thread == 0) {

            g_assert_cmpint(node, ==, 1);

        } else if (thread == 1) {

            g_assert_cmpint(node, ==, 0);

        } else {

            g_assert(false);

        }

    }



    QDECREF(resp);

    qtest_end();

    g_free(cli);

}
