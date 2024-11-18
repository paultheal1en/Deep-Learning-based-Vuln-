static void monitor_protocol_emitter(Monitor *mon, QObject *data,

                                     QError *err)

{

    QDict *qmp;



    trace_monitor_protocol_emitter(mon);



    if (!err) {

        /* success response */

        qmp = qdict_new();

        if (data) {

            qobject_incref(data);

            qdict_put_obj(qmp, "return", data);

        } else {

            /* return an empty QDict by default */

            qdict_put(qmp, "return", qdict_new());

        }

    } else {

        /* error response */

        qmp = build_qmp_error_dict(err);

    }



    if (mon->mc->id) {

        qdict_put_obj(qmp, "id", mon->mc->id);

        mon->mc->id = NULL;

    }



    monitor_json_emitter(mon, QOBJECT(qmp));

    QDECREF(qmp);

}
