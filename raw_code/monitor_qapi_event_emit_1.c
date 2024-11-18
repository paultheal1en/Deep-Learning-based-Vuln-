static void monitor_qapi_event_emit(QAPIEvent event, QDict *qdict)

{

    Monitor *mon;



    trace_monitor_protocol_event_emit(event, qdict);

    QLIST_FOREACH(mon, &mon_list, entry) {

        if (monitor_is_qmp(mon) && mon->qmp.in_command_mode) {

            monitor_json_emitter(mon, QOBJECT(qdict));

        }

    }

}
