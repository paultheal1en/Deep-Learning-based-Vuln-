static void spapr_numa_cpu(const void *data)

{

    char *cli;

    QDict *resp;

    QList *cpus;

    const QObject *e;



    cli = make_cli(data, "-smp 4,cores=4 "

        "-numa node,nodeid=0 -numa node,nodeid=1 "

        "-numa cpu,node-id=0,core-id=0 "

        "-numa cpu,node-id=0,core-id=1 "

        "-numa cpu,node-id=0,core-id=2 "

        "-numa cpu,node-id=1,core-id=3");

    qtest_start(cli);

    cpus = get_cpus(&resp);

    g_assert(cpus);



    while ((e = qlist_pop(cpus))) {

        QDict *cpu, *props;

        int64_t core, node;



        cpu = qobject_to_qdict(e);

        g_assert(qdict_haskey(cpu, "props"));

        props = qdict_get_qdict(cpu, "props");



        g_assert(qdict_haskey(props, "node-id"));

        node = qdict_get_int(props, "node-id");

        g_assert(qdict_haskey(props, "core-id"));

        core = qdict_get_int(props, "core-id");



        if (core >= 0 && core < 3) {

            g_assert_cmpint(node, ==, 0);

        } else if (core == 3) {

            g_assert_cmpint(node, ==, 1);

        } else {

            g_assert(false);

        }

    }



    QDECREF(resp);

    qtest_end();

    g_free(cli);

}
