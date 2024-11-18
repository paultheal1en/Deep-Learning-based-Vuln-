static void test_dispatch_cmd(void)

{

    QDict *req = qdict_new();

    QObject *resp;



    qdict_put_obj(req, "execute", QOBJECT(qstring_from_str("user_def_cmd")));



    resp = qmp_dispatch(QOBJECT(req));

    assert(resp != NULL);

    assert(!qdict_haskey(qobject_to_qdict(resp), "error"));

    g_print("\nresp: %s\n", qstring_get_str(qobject_to_json(resp)));



    qobject_decref(resp);

    QDECREF(req);

}
