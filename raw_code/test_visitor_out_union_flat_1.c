static void test_visitor_out_union_flat(TestOutputVisitorData *data,

                                        const void *unused)

{

    QObject *arg;

    QDict *qdict;



    UserDefFlatUnion *tmp = g_malloc0(sizeof(UserDefFlatUnion));

    tmp->enum1 = ENUM_ONE_VALUE1;

    tmp->string = g_strdup("str");

    tmp->u.value1 = g_malloc0(sizeof(UserDefA));

    tmp->integer = 41;

    tmp->u.value1->boolean = true;



    visit_type_UserDefFlatUnion(data->ov, NULL, &tmp, &error_abort);

    arg = qmp_output_get_qobject(data->qov);



    g_assert(qobject_type(arg) == QTYPE_QDICT);

    qdict = qobject_to_qdict(arg);



    g_assert_cmpstr(qdict_get_str(qdict, "enum1"), ==, "value1");

    g_assert_cmpstr(qdict_get_str(qdict, "string"), ==, "str");

    g_assert_cmpint(qdict_get_int(qdict, "integer"), ==, 41);

    g_assert_cmpint(qdict_get_bool(qdict, "boolean"), ==, true);



    qapi_free_UserDefFlatUnion(tmp);

    QDECREF(qdict);

}
