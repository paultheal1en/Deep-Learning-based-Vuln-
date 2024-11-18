static void test_visitor_in_bool(TestInputVisitorData *data,

                                 const void *unused)

{

    bool res = false;

    Visitor *v;



    v = visitor_input_test_init(data, "true");



    visit_type_bool(v, NULL, &res, &error_abort);

    g_assert_cmpint(res, ==, true);

}
