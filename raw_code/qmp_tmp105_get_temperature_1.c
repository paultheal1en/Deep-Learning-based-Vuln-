static int qmp_tmp105_get_temperature(const char *id)

{

    QDict *response;

    int ret;



    response = qmp("{ 'execute': 'qom-get', 'arguments': { 'path': '%s', "

                   "'property': 'temperature' } }", id);

    g_assert(qdict_haskey(response, "return"));

    ret = qdict_get_int(response, "return");

    QDECREF(response);

    return ret;

}
