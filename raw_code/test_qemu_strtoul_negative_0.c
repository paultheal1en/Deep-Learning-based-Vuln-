static void test_qemu_strtoul_negative(void)

{

    const char *str = "  \t -321";

    char f = 'X';

    const char *endptr = &f;

    unsigned long res = 999;

    int err;



    err = qemu_strtoul(str, &endptr, 0, &res);



    g_assert_cmpint(err, ==, 0);

    g_assert_cmpint(res, ==, -321ul);

    g_assert(endptr == str + strlen(str));

}
