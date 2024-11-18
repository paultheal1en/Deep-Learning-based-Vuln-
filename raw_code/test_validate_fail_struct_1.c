static void test_validate_fail_struct(TestInputVisitorData *data,

                                       const void *unused)

{

    TestStruct *p = NULL;

    Error *err = NULL;

    Visitor *v;



    v = validate_test_init(data, "{ 'integer': -42, 'boolean': true, 'string': 'foo', 'extra': 42 }");



    visit_type_TestStruct(v, &p, NULL, &err);

    g_assert(err);

    error_free(err);

    if (p) {

        g_free(p->string);

    }

    g_free(p);

}
