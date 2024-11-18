static void test_qemu_strtol_invalid(void)

{

    const char *str = "   xxxx  \t abc";

    char f = 'X';

    const char *endptr = &f;

    long res = 999;

    int err;



    err = qemu_strtol(str, &endptr, 0, &res);



    g_assert_cmpint(err, ==, 0);

    g_assert_cmpint(res, ==, 0);

    g_assert(endptr == str);

}
