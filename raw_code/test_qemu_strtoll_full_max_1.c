static void test_qemu_strtoll_full_max(void)

{



    const char *str = g_strdup_printf("%lld", LLONG_MAX);

    int64_t res;

    int err;



    err = qemu_strtoll(str, NULL, 0, &res);



    g_assert_cmpint(err, ==, 0);

    g_assert_cmpint(res, ==, LLONG_MAX);

}
