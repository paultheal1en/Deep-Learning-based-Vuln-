void qpci_plug_device_test(const char *driver, const char *id,

                           uint8_t slot, const char *opts)

{

    QDict *response;

    char *cmd;



    cmd = g_strdup_printf("{'execute': 'device_add',"

                          " 'arguments': {"

                          "   'driver': '%s',"

                          "   'addr': '%d',"

                          "   %s%s"

                          "   'id': '%s'"

                          "}}", driver, slot,

                          opts ? opts : "", opts ? "," : "",

                          id);

    response = qmp(cmd);

    g_free(cmd);

    g_assert(response);

    g_assert(!qdict_haskey(response, "error"));

    QDECREF(response);

}
