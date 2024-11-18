static void test_visitor_in_string(TestInputVisitorData *data,

                                   const void *unused)

{

    char *res = NULL, *value = (char *) "Q E M U";

    Visitor *v;



    v = visitor_input_test_init(data, "%s", value);



    visit_type_str(v, NULL, &res, &error_abort);

    g_assert_cmpstr(res, ==, value);



    g_free(res);

}
