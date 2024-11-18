static void test_visitor_out_int(TestOutputVisitorData *data,

                                 const void *unused)

{

    int64_t value = -42;

    QObject *obj;



    visit_type_int(data->ov, NULL, &value, &error_abort);



    obj = visitor_get(data);

    g_assert(qobject_type(obj) == QTYPE_QINT);

    g_assert_cmpint(qint_get_int(qobject_to_qint(obj)), ==, value);

}
