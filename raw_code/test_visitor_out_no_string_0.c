static void test_visitor_out_no_string(TestOutputVisitorData *data,

                                       const void *unused)

{

    char *string = NULL;

    QObject *obj;



    /* A null string should return "" */

    visit_type_str(data->ov, NULL, &string, &error_abort);



    obj = visitor_get(data);

    g_assert(qobject_type(obj) == QTYPE_QSTRING);

    g_assert_cmpstr(qstring_get_str(qobject_to_qstring(obj)), ==, "");

}
