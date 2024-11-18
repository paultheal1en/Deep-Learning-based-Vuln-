static void test_validate_fail_union_native_list(TestInputVisitorData *data,

                                                 const void *unused)

{

    UserDefNativeListUnion *tmp = NULL;

    Error *err = NULL;

    Visitor *v;



    v = validate_test_init(data,

                           "{ 'type': 'integer', 'data' : [ 'string' ] }");



    visit_type_UserDefNativeListUnion(v, NULL, &tmp, &err);

    error_free_or_abort(&err);

    g_assert(!tmp);

}
