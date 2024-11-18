static void test_qemu_strtoull_full_empty(void)

{

    const char *str = "";

    uint64_t res = 999;

    int err;



    err = qemu_strtoull(str, NULL, 0, &res);



    g_assert_cmpint(err, ==, 0);

    g_assert_cmpint(res, ==, 0);

}
