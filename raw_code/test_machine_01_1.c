static void test_machine(gconstpointer data)

{

    const char *machine = data;

    char *args;

    QDict *response;



    args = g_strdup_printf("-machine %s", machine);

    qtest_start(args);



    test_properties("/machine");



    response = qmp("{ 'execute': 'quit' }");

    g_assert(qdict_haskey(response, "return"));



    qtest_end();

    g_free(args);

}
