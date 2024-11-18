static void qmp_tmp105_set_temperature(const char *id, int value)

{

    QDict *response;



    response = qmp("{ 'execute': 'qom-set', 'arguments': { 'path': '%s', "

                   "'property': 'temperature', 'value': %d } }", id, value);

    g_assert(qdict_haskey(response, "return"));

    QDECREF(response);

}
