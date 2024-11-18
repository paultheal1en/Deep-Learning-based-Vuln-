static void test_qemu_strtosz_trailing(void)

{

    const char *str;

    char *endptr = NULL;

    int64_t res;



    str = "123xxx";

    res = qemu_strtosz_MiB(str, &endptr);

    g_assert_cmpint(res, ==, 123 * M_BYTE);

    g_assert(endptr == str + 3);



    res = qemu_strtosz(str, NULL);

    g_assert_cmpint(res, ==, -EINVAL);



    str = "1kiB";

    res = qemu_strtosz(str, &endptr);

    g_assert_cmpint(res, ==, 1024);

    g_assert(endptr == str + 2);



    res = qemu_strtosz(str, NULL);

    g_assert_cmpint(res, ==, -EINVAL);

}
