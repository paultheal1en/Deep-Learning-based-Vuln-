static sPAPREventLogEntry *rtas_event_log_dequeue(uint32_t event_mask)

{

    sPAPRMachineState *spapr = SPAPR_MACHINE(qdev_get_machine());

    sPAPREventLogEntry *entry = NULL;



    QTAILQ_FOREACH(entry, &spapr->pending_events, next) {

        const sPAPREventSource *source =

            rtas_event_log_to_source(spapr, entry->log_type);



        if (source->mask & event_mask) {

            break;

        }

    }



    if (entry) {

        QTAILQ_REMOVE(&spapr->pending_events, entry, next);

    }



    return entry;

}
