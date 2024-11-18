void spapr_clear_pending_events(sPAPRMachineState *spapr)

{

    sPAPREventLogEntry *entry = NULL;



    QTAILQ_FOREACH(entry, &spapr->pending_events, next) {

        QTAILQ_REMOVE(&spapr->pending_events, entry, next);

        g_free(entry->extended_log);

        g_free(entry);

    }

}
