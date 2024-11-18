QEMU_BUILD_BUG_ON(ARRAY_SIZE(monitor_event_names) != QEVENT_MAX)



/**

 * monitor_protocol_event(): Generate a Monitor event

 *

 * Event-specific data can be emitted through the (optional) 'data' parameter.

 */

void monitor_protocol_event(MonitorEvent event, QObject *data)

{

    QDict *qmp;

    const char *event_name;

    Monitor *mon;



    assert(event < QEVENT_MAX);



    event_name = monitor_event_names[event];

    assert(event_name != NULL);



    qmp = qdict_new();

    timestamp_put(qmp);

    qdict_put(qmp, "event", qstring_from_str(event_name));

    if (data) {

        qobject_incref(data);

        qdict_put_obj(qmp, "data", data);

    }



    QLIST_FOREACH(mon, &mon_list, entry) {

        if (monitor_ctrl_mode(mon) && qmp_cmd_mode(mon)) {

            monitor_json_emitter(mon, QOBJECT(qmp));

        }

    }

    QDECREF(qmp);

}
