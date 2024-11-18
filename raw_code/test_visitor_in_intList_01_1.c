static void test_visitor_in_intList(TestInputVisitorData *data,

                                    const void *unused)

{

    int64_t value[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20};

    int16List *res = NULL, *tmp;

    Error *err = NULL;

    Visitor *v;

    int i = 0;



    v = visitor_input_test_init(data, "1,2,0,2-4,20,5-9,1-8");



    visit_type_int16List(v, NULL, &res, &error_abort);

    tmp = res;

    while (i < sizeof(value) / sizeof(value[0])) {

        g_assert(tmp);

        g_assert_cmpint(tmp->value, ==, value[i++]);

        tmp = tmp->next;

    }

    g_assert(!tmp);



    tmp = res;

    while (tmp) {

        res = res->next;

        g_free(tmp);

        tmp = res;

    }



    visitor_input_teardown(data, unused);



    v = visitor_input_test_init(data, "not an int list");



    visit_type_int16List(v, NULL, &res, &err);

    /* FIXME fix the visitor, then error_free_or_abort(&err) here */

}
