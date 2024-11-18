static void test_visitor_out_any(TestOutputVisitorData *data,

                                 const void *unused)

{

    QObject *qobj;

    QInt *qint;

    QBool *qbool;

    QString *qstring;

    QDict *qdict;

    QObject *obj;



    qobj = QOBJECT(qint_from_int(-42));

    visit_type_any(data->ov, NULL, &qobj, &error_abort);

    obj = visitor_get(data);

    g_assert(qobject_type(obj) == QTYPE_QINT);

    g_assert_cmpint(qint_get_int(qobject_to_qint(obj)), ==, -42);

    qobject_decref(qobj);



    visitor_reset(data);

    qdict = qdict_new();

    qdict_put(qdict, "integer", qint_from_int(-42));

    qdict_put(qdict, "boolean", qbool_from_bool(true));

    qdict_put(qdict, "string", qstring_from_str("foo"));

    qobj = QOBJECT(qdict);

    visit_type_any(data->ov, NULL, &qobj, &error_abort);

    qobject_decref(qobj);

    qdict = qobject_to_qdict(visitor_get(data));

    g_assert(qdict);

    qobj = qdict_get(qdict, "integer");

    g_assert(qobj);

    qint = qobject_to_qint(qobj);

    g_assert(qint);

    g_assert_cmpint(qint_get_int(qint), ==, -42);

    qobj = qdict_get(qdict, "boolean");

    g_assert(qobj);

    qbool = qobject_to_qbool(qobj);

    g_assert(qbool);

    g_assert(qbool_get_bool(qbool) == true);

    qstring = qobject_to_qstring(qdict_get(qdict, "string"));

    g_assert(qstring);

    g_assert_cmpstr(qstring_get_str(qstring), ==, "foo");

}
