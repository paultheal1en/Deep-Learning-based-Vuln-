void hmp_dump_guest_memory(Monitor *mon, const QDict *qdict)

{

    Error *errp = NULL;

    int paging = qdict_get_try_bool(qdict, "paging", 0);

    const char *file = qdict_get_str(qdict, "filename");

    bool has_begin = qdict_haskey(qdict, "begin");

    bool has_length = qdict_haskey(qdict, "length");

    int64_t begin = 0;

    int64_t length = 0;

    char *prot;



    if (has_begin) {

        begin = qdict_get_int(qdict, "begin");

    }

    if (has_length) {

        length = qdict_get_int(qdict, "length");

    }



    prot = g_strconcat("file:", file, NULL);



    qmp_dump_guest_memory(paging, prot, has_begin, begin, has_length, length,

                          &errp);

    hmp_handle_error(mon, &errp);

    g_free(prot);

}
