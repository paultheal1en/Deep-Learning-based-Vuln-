static void test_visitor_in_wrong_type(TestInputVisitorData *data,

                                       const void *unused)

{

    TestStruct *p = NULL;

    Visitor *v;

    strList *q = NULL;

    int64_t i;

    Error *err = NULL;



    /* Make sure arrays and structs cannot be confused */



    v = visitor_input_test_init(data, "[]");

    visit_type_TestStruct(v, NULL, &p, &err);

    error_free_or_abort(&err);

    g_assert(!p);



    v = visitor_input_test_init(data, "{}");

    visit_type_strList(v, NULL, &q, &err);

    error_free_or_abort(&err);

    assert(!q);



    /* Make sure primitives and struct cannot be confused */



    v = visitor_input_test_init(data, "1");

    visit_type_TestStruct(v, NULL, &p, &err);

    error_free_or_abort(&err);

    g_assert(!p);



    v = visitor_input_test_init(data, "{}");

    visit_type_int(v, NULL, &i, &err);

    error_free_or_abort(&err);



    /* Make sure primitives and arrays cannot be confused */



    v = visitor_input_test_init(data, "1");

    visit_type_strList(v, NULL, &q, &err);

    error_free_or_abort(&err);

    assert(!q);



    v = visitor_input_test_init(data, "[]");

    visit_type_int(v, NULL, &i, &err);

    error_free_or_abort(&err);

}
