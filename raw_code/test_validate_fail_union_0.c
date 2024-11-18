static void test_validate_fail_union(TestInputVisitorData *data,

                                      const void *unused)

{

    UserDefUnion *tmp = NULL;

    Error *err = NULL;

    Visitor *v;



    v = validate_test_init(data, "{ 'type': 'b', 'data' : { 'integer': 42 } }");



    visit_type_UserDefUnion(v, &tmp, NULL, &err);

    g_assert(err);

    qapi_free_UserDefUnion(tmp);

}
