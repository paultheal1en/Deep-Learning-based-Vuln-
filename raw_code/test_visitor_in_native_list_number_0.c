static void test_visitor_in_native_list_number(TestInputVisitorData *data,

                                               const void *unused)

{

    UserDefNativeListUnion *cvalue = NULL;

    numberList *elem = NULL;

    Error *err = NULL;

    Visitor *v;

    GString *gstr_list = g_string_new("");

    GString *gstr_union = g_string_new("");

    int i;



    for (i = 0; i < 32; i++) {

        g_string_append_printf(gstr_list, "%f", (double)i / 3);

        if (i != 31) {

            g_string_append(gstr_list, ", ");

        }

    }

    g_string_append_printf(gstr_union,  "{ 'type': 'number', 'data': [ %s ] }",

                           gstr_list->str);

    v = visitor_input_test_init_raw(data,  gstr_union->str);



    visit_type_UserDefNativeListUnion(v, &cvalue, NULL, &err);

    g_assert(err == NULL);

    g_assert(cvalue != NULL);

    g_assert_cmpint(cvalue->type, ==, USER_DEF_NATIVE_LIST_UNION_KIND_NUMBER);



    for (i = 0, elem = cvalue->u.number; elem; elem = elem->next, i++) {

        GString *double_expected = g_string_new("");

        GString *double_actual = g_string_new("");



        g_string_printf(double_expected, "%.6f", (double)i / 3);

        g_string_printf(double_actual, "%.6f", elem->value);

        g_assert_cmpstr(double_expected->str, ==, double_actual->str);



        g_string_free(double_expected, true);

        g_string_free(double_actual, true);

    }



    g_string_free(gstr_union, true);

    g_string_free(gstr_list, true);

    qapi_free_UserDefNativeListUnion(cvalue);

}
