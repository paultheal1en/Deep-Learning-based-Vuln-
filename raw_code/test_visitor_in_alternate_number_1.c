static void test_visitor_in_alternate_number(TestInputVisitorData *data,

                                             const void *unused)

{

    Visitor *v;

    Error *err = NULL;

    AltStrBool *asb;

    AltStrNum *asn;

    AltNumStr *ans;

    AltStrInt *asi;

    AltIntNum *ain;

    AltNumInt *ani;



    /* Parsing an int */



    v = visitor_input_test_init(data, "42");

    visit_type_AltStrBool(v, &asb, NULL, &err);

    error_free_or_abort(&err);

    qapi_free_AltStrBool(asb);



    /* FIXME: Order of alternate should not affect semantics; asn should

     * parse the same as ans */

    v = visitor_input_test_init(data, "42");

    visit_type_AltStrNum(v, &asn, NULL, &err);

    /* FIXME g_assert_cmpint(asn->type, == ALT_STR_NUM_KIND_N); */

    /* FIXME g_assert_cmpfloat(asn->u.n, ==, 42); */

    error_free_or_abort(&err);

    qapi_free_AltStrNum(asn);



    v = visitor_input_test_init(data, "42");

    visit_type_AltNumStr(v, &ans, NULL, &error_abort);

    g_assert_cmpint(ans->type, ==, ALT_NUM_STR_KIND_N);

    g_assert_cmpfloat(ans->u.n, ==, 42);

    qapi_free_AltNumStr(ans);



    v = visitor_input_test_init(data, "42");

    visit_type_AltStrInt(v, &asi, NULL, &error_abort);

    g_assert_cmpint(asi->type, ==, ALT_STR_INT_KIND_I);

    g_assert_cmpint(asi->u.i, ==, 42);

    qapi_free_AltStrInt(asi);



    v = visitor_input_test_init(data, "42");

    visit_type_AltIntNum(v, &ain, NULL, &error_abort);

    g_assert_cmpint(ain->type, ==, ALT_INT_NUM_KIND_I);

    g_assert_cmpint(ain->u.i, ==, 42);

    qapi_free_AltIntNum(ain);



    v = visitor_input_test_init(data, "42");

    visit_type_AltNumInt(v, &ani, NULL, &error_abort);

    g_assert_cmpint(ani->type, ==, ALT_NUM_INT_KIND_I);

    g_assert_cmpint(ani->u.i, ==, 42);

    qapi_free_AltNumInt(ani);



    /* Parsing a double */



    v = visitor_input_test_init(data, "42.5");

    visit_type_AltStrBool(v, &asb, NULL, &err);

    error_free_or_abort(&err);

    qapi_free_AltStrBool(asb);



    v = visitor_input_test_init(data, "42.5");

    visit_type_AltStrNum(v, &asn, NULL, &error_abort);

    g_assert_cmpint(asn->type, ==, ALT_STR_NUM_KIND_N);

    g_assert_cmpfloat(asn->u.n, ==, 42.5);

    qapi_free_AltStrNum(asn);



    v = visitor_input_test_init(data, "42.5");

    visit_type_AltNumStr(v, &ans, NULL, &error_abort);

    g_assert_cmpint(ans->type, ==, ALT_NUM_STR_KIND_N);

    g_assert_cmpfloat(ans->u.n, ==, 42.5);

    qapi_free_AltNumStr(ans);



    v = visitor_input_test_init(data, "42.5");

    visit_type_AltStrInt(v, &asi, NULL, &err);

    error_free_or_abort(&err);

    qapi_free_AltStrInt(asi);



    v = visitor_input_test_init(data, "42.5");

    visit_type_AltIntNum(v, &ain, NULL, &error_abort);

    g_assert_cmpint(ain->type, ==, ALT_INT_NUM_KIND_N);

    g_assert_cmpfloat(ain->u.n, ==, 42.5);

    qapi_free_AltIntNum(ain);



    v = visitor_input_test_init(data, "42.5");

    visit_type_AltNumInt(v, &ani, NULL, &error_abort);

    g_assert_cmpint(ani->type, ==, ALT_NUM_INT_KIND_N);

    g_assert_cmpfloat(ani->u.n, ==, 42.5);

    qapi_free_AltNumInt(ani);

}
