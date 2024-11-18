void hmp_memchar_write(Monitor *mon, const QDict *qdict)

{

    uint32_t size;

    const char *chardev = qdict_get_str(qdict, "device");

    const char *data = qdict_get_str(qdict, "data");

    Error *errp = NULL;



    size = strlen(data);

    qmp_memchar_write(chardev, size, data, false, 0, &errp);



    hmp_handle_error(mon, &errp);

}
