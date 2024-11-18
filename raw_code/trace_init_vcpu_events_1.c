void trace_init_vcpu_events(void)

{

    TraceEvent *ev = NULL;

    while ((ev = trace_event_pattern("*", ev)) != NULL) {

        if (trace_event_is_vcpu(ev) &&

            trace_event_get_state_static(ev) &&

            trace_event_get_state_dynamic(ev)) {

            TraceEventID id = trace_event_get_id(ev);

            /* check preconditions */

            assert(trace_events_dstate[id] == 1);

            /* disable early-init state ... */

            trace_events_dstate[id] = 0;

            trace_events_enabled_count--;

            /* ... and properly re-enable */

            trace_event_set_state_dynamic(ev, true);

        }

    }

}
