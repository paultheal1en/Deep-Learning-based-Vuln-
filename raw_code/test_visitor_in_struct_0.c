static void test_visitor_in_struct(TestInputVisitorData *data,

                                   const void *unused)

{

    TestStruct *p = NULL;

    Visitor *v;



    v = visitor_input_test_init(data, "{ 'integer': -42, 'boolean': true, 'string': 'foo' }");



    visit_type_TestStruct(v, NULL, &p, &error_abort);

    g_assert_cmpint(p->integer, ==, -42);

    g_assert(p->boolean == true);

    g_assert_cmpstr(p->string, ==, "foo");



    g_free(p->string);

    g_free(p);

}
