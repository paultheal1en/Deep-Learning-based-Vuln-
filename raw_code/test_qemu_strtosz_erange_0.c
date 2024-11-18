static void test_qemu_strtosz_erange(void)

{

    const char *str = "10E";

    char *endptr = NULL;

    int64_t res;



    res = qemu_strtosz(str, &endptr);

    g_assert_cmpint(res, ==, -ERANGE);

    g_assert(endptr == str + 3);

}
