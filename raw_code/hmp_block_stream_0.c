void hmp_block_stream(Monitor *mon, const QDict *qdict)

{

    Error *error = NULL;

    const char *device = qdict_get_str(qdict, "device");

    const char *base = qdict_get_try_str(qdict, "base");



    qmp_block_stream(device, base != NULL, base, &error);



    hmp_handle_error(mon, &error);

}
