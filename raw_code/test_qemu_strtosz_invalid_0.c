static void test_qemu_strtosz_invalid(void)

{

    const char *str;

    char *endptr = NULL;

    int64_t res;



    str = "";

    res = qemu_strtosz(str, &endptr);

    g_assert_cmpint(res, ==, -EINVAL);

    g_assert(endptr == str);



    str = " \t ";

    res = qemu_strtosz(str, &endptr);

    g_assert_cmpint(res, ==, -EINVAL);

    g_assert(endptr == str);



    str = "crap";

    res = qemu_strtosz(str, &endptr);

    g_assert_cmpint(res, ==, -EINVAL);

    g_assert(endptr == str);

}
