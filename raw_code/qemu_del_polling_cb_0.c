void qemu_del_polling_cb(PollingFunc *func, void *opaque)

{

    PollingEntry **ppe, *pe;

    for(ppe = &first_polling_entry; *ppe != NULL; ppe = &(*ppe)->next) {

        pe = *ppe;

        if (pe->func == func && pe->opaque == opaque) {

            *ppe = pe->next;

            g_free(pe);

            break;

        }

    }

}
