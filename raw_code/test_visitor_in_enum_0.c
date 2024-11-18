static void test_visitor_in_enum(TestInputVisitorData *data,

                                 const void *unused)

{

    Error *err = NULL;

    Visitor *v;

    EnumOne i;



    for (i = 0; EnumOne_lookup[i]; i++) {

        EnumOne res = -1;



        v = visitor_input_test_init(data, "%s", EnumOne_lookup[i]);



        visit_type_EnumOne(v, &res, NULL, &err);

        g_assert(!err);

        g_assert_cmpint(i, ==, res);

    }

}
