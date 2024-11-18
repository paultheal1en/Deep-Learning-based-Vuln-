static void test_visitor_in_fail_list(TestInputVisitorData *data,

                                      const void *unused)

{

    int64_t i64 = -1;

    Visitor *v;



    /* Unvisited list tail */



    v = visitor_input_test_init(data, "[ 1, 2, 3 ]");



    visit_start_list(v, NULL, NULL, 0, &error_abort);

    visit_type_int(v, NULL, &i64, &error_abort);

    g_assert_cmpint(i64, ==, 1);

    visit_type_int(v, NULL, &i64, &error_abort);

    g_assert_cmpint(i64, ==, 2);

    visit_end_list(v, NULL);

    /* BUG: unvisited tail not reported; actually not reportable by design */

}
