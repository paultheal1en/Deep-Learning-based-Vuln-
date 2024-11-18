static void test_validate_struct(TestInputVisitorData *data,

                                  const void *unused)

{

    TestStruct *p = NULL;

    Visitor *v;



    v = validate_test_init(data, "{ 'integer': -42, 'boolean': true, 'string': 'foo' }");



    visit_type_TestStruct(v, NULL, &p, &error_abort);

    g_free(p->string);

    g_free(p);

}
