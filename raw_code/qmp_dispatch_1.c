QObject *qmp_dispatch(QObject *request)

{

    Error *err = NULL;

    QObject *ret;

    QDict *rsp;



    ret = do_qmp_dispatch(request, &err);



    rsp = qdict_new();

    if (err) {

        qdict_put_obj(rsp, "error", error_get_qobject(err));

        error_free(err);

    } else if (ret) {

        qdict_put_obj(rsp, "return", ret);

    } else {

        QDECREF(rsp);

        return NULL;

    }



    return QOBJECT(rsp);

}
