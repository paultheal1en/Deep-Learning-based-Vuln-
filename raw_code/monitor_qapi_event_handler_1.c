static void monitor_qapi_event_handler(void *opaque)

{

    MonitorQAPIEventState *evstate = opaque;

    MonitorQAPIEventConf *evconf = &monitor_qapi_event_conf[evstate->event];



    trace_monitor_protocol_event_handler(evstate->event, evstate->qdict);

    qemu_mutex_lock(&monitor_lock);



    if (evstate->qdict) {

        int64_t now = qemu_clock_get_ns(QEMU_CLOCK_REALTIME);



        monitor_qapi_event_emit(evstate->event, evstate->qdict);

        QDECREF(evstate->qdict);

        evstate->qdict = NULL;

        timer_mod_ns(evstate->timer, now + evconf->rate);

    } else {

        g_hash_table_remove(monitor_qapi_event_state, evstate);

        QDECREF(evstate->data);

        timer_free(evstate->timer);

        g_free(evstate);

    }



    qemu_mutex_unlock(&monitor_lock);

}
