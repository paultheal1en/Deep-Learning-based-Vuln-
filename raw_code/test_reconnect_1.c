static void test_reconnect(void)

{

    gchar *path = g_strdup_printf("/%s/vhost-user/reconnect/subprocess",

                                  qtest_get_arch());

    g_test_trap_subprocess(path, 0, 0);

    g_test_trap_assert_passed();


}