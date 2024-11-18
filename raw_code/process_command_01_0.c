static void process_command(GAState *s, QDict *req)

{

    QObject *rsp = NULL;

    int ret;



    g_assert(req);

    g_debug("processing command");

    rsp = qmp_dispatch(QOBJECT(req));

    if (rsp) {

        ret = send_response(s, rsp);

        if (ret) {

            g_warning("error sending response: %s", strerror(ret));

        }

        qobject_decref(rsp);

    } else {

        g_warning("error getting response");

    }

}
