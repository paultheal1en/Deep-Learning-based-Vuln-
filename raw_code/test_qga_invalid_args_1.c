static void test_qga_invalid_args(gconstpointer fix)

{

    const TestFixture *fixture = fix;

    QDict *ret, *error;

    const gchar *class, *desc;



    ret = qmp_fd(fixture->fd, "{'execute': 'guest-ping', "

                 "'arguments': {'foo': 42 }}");

    g_assert_nonnull(ret);



    error = qdict_get_qdict(ret, "error");

    class = qdict_get_try_str(error, "class");

    desc = qdict_get_try_str(error, "desc");



    g_assert_cmpstr(class, ==, "GenericError");

    g_assert_cmpstr(desc, ==, "QMP input object member 'foo' is unexpected");



    QDECREF(ret);

}
