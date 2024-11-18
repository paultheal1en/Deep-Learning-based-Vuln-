void monitor_protocol_event(MonitorEvent event, QObject *data)

{

    QDict *qmp;

    const char *event_name;

    Monitor *mon;



    assert(event < QEVENT_MAX);



    switch (event) {

        case QEVENT_DEBUG:

            event_name = "DEBUG";

            break;

        case QEVENT_SHUTDOWN:

            event_name = "SHUTDOWN";

            break;

        case QEVENT_RESET:

            event_name = "RESET";

            break;

        case QEVENT_POWERDOWN:

            event_name = "POWERDOWN";

            break;

        case QEVENT_STOP:

            event_name = "STOP";

            break;

        case QEVENT_VNC_CONNECTED:

            event_name = "VNC_CONNECTED";

            break;

        case QEVENT_VNC_INITIALIZED:

            event_name = "VNC_INITIALIZED";

            break;

        case QEVENT_VNC_DISCONNECTED:

            event_name = "VNC_DISCONNECTED";

            break;

        default:

            abort();

            break;

    }



    qmp = qdict_new();

    timestamp_put(qmp);

    qdict_put(qmp, "event", qstring_from_str(event_name));

    if (data) {

        qobject_incref(data);

        qdict_put_obj(qmp, "data", data);

    }



    QLIST_FOREACH(mon, &mon_list, entry) {

        if (!monitor_ctrl_mode(mon))

            return;



        monitor_json_emitter(mon, QOBJECT(qmp));

    }

    QDECREF(qmp);

}
