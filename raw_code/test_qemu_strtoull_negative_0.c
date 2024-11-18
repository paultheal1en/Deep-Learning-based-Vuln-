static void test_qemu_strtoull_negative(void)

{

    const char *str = "  \t -321";

    char f = 'X';

    const char *endptr = &f;

    uint64_t res = 999;

    int err;



    err = qemu_strtoull(str, &endptr, 0, &res);



    g_assert_cmpint(err, ==, 0);

    g_assert_cmpint(res, ==, -321);

    g_assert(endptr == str + strlen(str));

}
