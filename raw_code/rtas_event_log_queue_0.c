static void rtas_event_log_queue(int log_type, void *data, bool exception)

{

    sPAPRMachineState *spapr = SPAPR_MACHINE(qdev_get_machine());

    sPAPREventLogEntry *entry = g_new(sPAPREventLogEntry, 1);



    g_assert(data);

    entry->log_type = log_type;

    entry->exception = exception;

    entry->data = data;

    QTAILQ_INSERT_TAIL(&spapr->pending_events, entry, next);

}
