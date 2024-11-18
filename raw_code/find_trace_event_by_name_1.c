static TraceEvent* find_trace_event_by_name(const char *tname)

{

    unsigned int i;



    if (!tname) {

        return NULL;

    }



    for (i = 0; i < NR_TRACE_EVENTS; i++) {

        if (!strcmp(trace_list[i].tp_name, tname)) {

            return &trace_list[i];

        }

    }

    return NULL; /* indicates end of list reached without a match */

}
