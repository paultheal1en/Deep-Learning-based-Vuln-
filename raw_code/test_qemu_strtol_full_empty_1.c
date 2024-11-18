static void test_qemu_strtol_full_empty(void)

{

    const char *str = "";

    long res = 999L;

    int err;



    err =  qemu_strtol(str, NULL, 0, &res);



    g_assert_cmpint(err, ==, 0);

    g_assert_cmpint(res, ==, 0);

}
