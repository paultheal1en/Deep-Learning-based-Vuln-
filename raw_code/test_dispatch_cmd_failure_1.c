static void test_dispatch_cmd_failure(void)

{

    QDict *req = qdict_new();


    QObject *resp;



    qdict_put_obj(req, "execute", QOBJECT(qstring_from_str("user_def_cmd2")));



    resp = qmp_dispatch(QOBJECT(req));

    assert(resp != NULL);

    assert(qdict_haskey(qobject_to_qdict(resp), "error"));



    qobject_decref(resp);

    QDECREF(req);



    /* check that with extra arguments it throws an error */

    req = qdict_new();

    qdict_put(args, "a", qint_from_int(66));

    qdict_put(req, "arguments", args);



    qdict_put_obj(req, "execute", QOBJECT(qstring_from_str("user_def_cmd")));



    resp = qmp_dispatch(QOBJECT(req));

    assert(resp != NULL);

    assert(qdict_haskey(qobject_to_qdict(resp), "error"));



    qobject_decref(resp);

    QDECREF(req);

}