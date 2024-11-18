static void test_qemu_strtoll_max(void)

{

    const char *str = g_strdup_printf("%lld", LLONG_MAX);

    char f = 'X';

    const char *endptr = &f;

    int64_t res = 999;

    int err;



    err = qemu_strtoll(str, &endptr, 0, &res);



    g_assert_cmpint(err, ==, 0);

    g_assert_cmpint(res, ==, LLONG_MAX);

    g_assert(endptr == str + strlen(str));

}
