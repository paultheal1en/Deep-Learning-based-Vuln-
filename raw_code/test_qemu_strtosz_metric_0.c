static void test_qemu_strtosz_metric(void)

{

    const char *str = "12345k";

    char *endptr = NULL;

    int64_t res;



    res = qemu_strtosz_metric(str, &endptr);

    g_assert_cmpint(res, ==, 12345000);

    g_assert(endptr == str + 6);

}
