int qemu_add_polling_cb(PollingFunc *func, void *opaque)

{

    PollingEntry **ppe, *pe;

    pe = g_malloc0(sizeof(PollingEntry));

    pe->func = func;

    pe->opaque = opaque;

    for(ppe = &first_polling_entry; *ppe != NULL; ppe = &(*ppe)->next);

    *ppe = pe;

    return 0;

}
