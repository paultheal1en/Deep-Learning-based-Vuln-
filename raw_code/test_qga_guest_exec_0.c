static void test_qga_guest_exec(gconstpointer fix)

{

    const TestFixture *fixture = fix;

    QDict *ret, *val;

    const gchar *out;

    guchar *decoded;

    int64_t pid, now, exitcode;

    gsize len;

    bool exited;



    /* exec 'echo foo bar' */

    ret = qmp_fd(fixture->fd, "{'execute': 'guest-exec', 'arguments': {"

                 " 'path': '/bin/echo', 'arg': [ '-n', '\" test_str \"' ],"

                 " 'capture-output': true } }");

    g_assert_nonnull(ret);

    qmp_assert_no_error(ret);

    val = qdict_get_qdict(ret, "return");

    pid = qdict_get_int(val, "pid");

    g_assert_cmpint(pid, >, 0);

    QDECREF(ret);



    /* wait for completion */

    now = g_get_monotonic_time();

    do {

        ret = qmp_fd(fixture->fd, "{'execute': 'guest-exec-status',"

                     " 'arguments': { 'pid': %" PRId64 "  } }", pid);

        g_assert_nonnull(ret);

        val = qdict_get_qdict(ret, "return");

        exited = qdict_get_bool(val, "exited");

        if (!exited) {

            QDECREF(ret);

        }

    } while (!exited &&

             g_get_monotonic_time() < now + 5 * G_TIME_SPAN_SECOND);

    g_assert(exited);



    /* check stdout */

    exitcode = qdict_get_int(val, "exitcode");

    g_assert_cmpint(exitcode, ==, 0);

    out = qdict_get_str(val, "out-data");

    decoded = g_base64_decode(out, &len);

    g_assert_cmpint(len, ==, 12);

    g_assert_cmpstr((char *)decoded, ==, "\" test_str \"");

    g_free(decoded);

    QDECREF(ret);

}
