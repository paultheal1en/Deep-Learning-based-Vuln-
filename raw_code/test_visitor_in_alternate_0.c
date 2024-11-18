static void test_visitor_in_alternate(TestInputVisitorData *data,

                                      const void *unused)

{

    Visitor *v;

    Error *err = NULL;

    UserDefAlternate *tmp;

    WrapAlternate *wrap;



    v = visitor_input_test_init(data, "42");

    visit_type_UserDefAlternate(v, NULL, &tmp, &error_abort);

    g_assert_cmpint(tmp->type, ==, QTYPE_QINT);

    g_assert_cmpint(tmp->u.i, ==, 42);

    qapi_free_UserDefAlternate(tmp);



    v = visitor_input_test_init(data, "'string'");

    visit_type_UserDefAlternate(v, NULL, &tmp, &error_abort);

    g_assert_cmpint(tmp->type, ==, QTYPE_QSTRING);

    g_assert_cmpstr(tmp->u.s, ==, "string");

    qapi_free_UserDefAlternate(tmp);



    v = visitor_input_test_init(data, "{'integer':1, 'string':'str', "

                                "'enum1':'value1', 'boolean':true}");

    visit_type_UserDefAlternate(v, NULL, &tmp, &error_abort);

    g_assert_cmpint(tmp->type, ==, QTYPE_QDICT);

    g_assert_cmpint(tmp->u.udfu->integer, ==, 1);

    g_assert_cmpstr(tmp->u.udfu->string, ==, "str");

    g_assert_cmpint(tmp->u.udfu->enum1, ==, ENUM_ONE_VALUE1);

    g_assert_cmpint(tmp->u.udfu->u.value1->boolean, ==, true);

    g_assert_cmpint(tmp->u.udfu->u.value1->has_a_b, ==, false);

    qapi_free_UserDefAlternate(tmp);



    v = visitor_input_test_init(data, "false");

    visit_type_UserDefAlternate(v, NULL, &tmp, &err);

    error_free_or_abort(&err);

    qapi_free_UserDefAlternate(tmp);



    v = visitor_input_test_init(data, "{ 'alt': 42 }");

    visit_type_WrapAlternate(v, NULL, &wrap, &error_abort);

    g_assert_cmpint(wrap->alt->type, ==, QTYPE_QINT);

    g_assert_cmpint(wrap->alt->u.i, ==, 42);

    qapi_free_WrapAlternate(wrap);



    v = visitor_input_test_init(data, "{ 'alt': 'string' }");

    visit_type_WrapAlternate(v, NULL, &wrap, &error_abort);

    g_assert_cmpint(wrap->alt->type, ==, QTYPE_QSTRING);

    g_assert_cmpstr(wrap->alt->u.s, ==, "string");

    qapi_free_WrapAlternate(wrap);



    v = visitor_input_test_init(data, "{ 'alt': {'integer':1, 'string':'str', "

                                "'enum1':'value1', 'boolean':true} }");

    visit_type_WrapAlternate(v, NULL, &wrap, &error_abort);

    g_assert_cmpint(wrap->alt->type, ==, QTYPE_QDICT);

    g_assert_cmpint(wrap->alt->u.udfu->integer, ==, 1);

    g_assert_cmpstr(wrap->alt->u.udfu->string, ==, "str");

    g_assert_cmpint(wrap->alt->u.udfu->enum1, ==, ENUM_ONE_VALUE1);

    g_assert_cmpint(wrap->alt->u.udfu->u.value1->boolean, ==, true);

    g_assert_cmpint(wrap->alt->u.udfu->u.value1->has_a_b, ==, false);

    qapi_free_WrapAlternate(wrap);

}
