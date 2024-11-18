static void test_visitor_in_int_overflow(TestInputVisitorData *data,

                                         const void *unused)

{

    int64_t res = 0;

    Error *err = NULL;

    Visitor *v;



    /* this will overflow a Qint/int64, so should be deserialized into

     * a QFloat/double field instead, leading to an error if we pass it

     * to visit_type_int. confirm this.

     */

    v = visitor_input_test_init(data, "%f", DBL_MAX);



    visit_type_int(v, &res, NULL, &err);

    g_assert(err);

    error_free(err);

}
