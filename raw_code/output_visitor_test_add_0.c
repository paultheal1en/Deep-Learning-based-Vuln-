static void output_visitor_test_add(const char *testpath,

                                    TestOutputVisitorData *data,

                                    void (*test_func)(TestOutputVisitorData *data, const void *user_data))

{

    g_test_add(testpath, TestOutputVisitorData, data, visitor_output_setup,

               test_func, visitor_output_teardown);

}
