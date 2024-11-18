static void test_visitor_out_enum(TestOutputVisitorData *data,

                                  const void *unused)

{

    QObject *obj;

    EnumOne i;



    for (i = 0; i < ENUM_ONE__MAX; i++) {

        visit_type_EnumOne(data->ov, "unused", &i, &error_abort);



        obj = visitor_get(data);

        g_assert(qobject_type(obj) == QTYPE_QSTRING);

        g_assert_cmpstr(qstring_get_str(qobject_to_qstring(obj)), ==,

                        EnumOne_lookup[i]);

        visitor_reset(data);

    }

}
