static void test_visitor_in_intList(TestInputVisitorData *data,
                                    const void *unused)
{
    /* Note: the visitor *sorts* ranges *unsigned* */
    int64_t expect1[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20 };
    int64_t expect2[] = { 32767, -32768, -32767 };
    int64_t expect3[] = { INT64_MAX, INT64_MIN };
    uint64_t expect4[] = { UINT64_MAX };
    Error *err = NULL;
    int64List *res = NULL;
    int64List *tail;
    Visitor *v;
    int64_t val;
    /* Valid lists */
    v = visitor_input_test_init(data, "1,2,0,2-4,20,5-9,1-8");
    check_ilist(v, expect1, ARRAY_SIZE(expect1));
    v = visitor_input_test_init(data, "32767,-32768--32767");
    check_ilist(v, expect2, ARRAY_SIZE(expect2));
    v = visitor_input_test_init(data,
                                "-9223372036854775808,9223372036854775807");
    check_ilist(v, expect3, ARRAY_SIZE(expect3));
    v = visitor_input_test_init(data, "18446744073709551615");
    check_ulist(v, expect4, ARRAY_SIZE(expect4));
    /* Empty list is invalid (weird) */
    v = visitor_input_test_init(data, "");
    visit_type_int64List(v, NULL, &res, &err);
    error_free_or_abort(&err);
    /* Not a list */
    v = visitor_input_test_init(data, "not an int list");
    visit_type_int64List(v, NULL, &res, &err);
    error_free_or_abort(&err);
    g_assert(!res);
    /* Unvisited list tail */
    v = visitor_input_test_init(data, "0,2-3");
    /* Would be simpler if the visitor genuinely supported virtual walks */
    visit_type_int64(v, NULL, &tail->value, &error_abort);
    tail = (int64List *)visit_next_list(v, (GenericList *)tail, sizeof(*res));
    g_assert(tail);
    visit_type_int64(v, NULL, &tail->value, &error_abort);
    g_assert_cmpint(tail->value, ==, 2);
    tail = (int64List *)visit_next_list(v, (GenericList *)tail, sizeof(*res));
    g_assert(tail);
    visit_check_list(v, &err);
    error_free_or_abort(&err);
}