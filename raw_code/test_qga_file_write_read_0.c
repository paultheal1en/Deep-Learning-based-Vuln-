static void test_qga_file_write_read(gconstpointer fix)

{

    const TestFixture *fixture = fix;

    const unsigned char helloworld[] = "Hello World!\n";

    const char *b64;

    gchar *cmd, *enc;

    QDict *ret, *val;

    int64_t id, eof;

    gsize count;



    /* open */

    ret = qmp_fd(fixture->fd, "{'execute': 'guest-file-open',"

                 " 'arguments': { 'path': 'foo', 'mode': 'w+' } }");

    g_assert_nonnull(ret);

    qmp_assert_no_error(ret);

    id = qdict_get_int(ret, "return");

    QDECREF(ret);



    enc = g_base64_encode(helloworld, sizeof(helloworld));

    /* write */

    cmd = g_strdup_printf("{'execute': 'guest-file-write',"

                          " 'arguments': { 'handle': %" PRId64 ","

                          " 'buf-b64': '%s' } }", id, enc);

    ret = qmp_fd(fixture->fd, cmd);

    g_assert_nonnull(ret);

    qmp_assert_no_error(ret);



    val = qdict_get_qdict(ret, "return");

    count = qdict_get_int(val, "count");

    eof = qdict_get_bool(val, "eof");

    g_assert_cmpint(count, ==, sizeof(helloworld));

    g_assert_cmpint(eof, ==, 0);

    QDECREF(ret);

    g_free(cmd);



    /* read (check implicit flush) */

    cmd = g_strdup_printf("{'execute': 'guest-file-read',"

                          " 'arguments': { 'handle': %" PRId64 "} }",

                          id);

    ret = qmp_fd(fixture->fd, cmd);

    val = qdict_get_qdict(ret, "return");

    count = qdict_get_int(val, "count");

    eof = qdict_get_bool(val, "eof");

    b64 = qdict_get_str(val, "buf-b64");

    g_assert_cmpint(count, ==, 0);

    g_assert(eof);

    g_assert_cmpstr(b64, ==, "");

    QDECREF(ret);

    g_free(cmd);



    /* seek to 0 */

    cmd = g_strdup_printf("{'execute': 'guest-file-seek',"

                          " 'arguments': { 'handle': %" PRId64 ", "

                          " 'offset': %d, 'whence': %d } }",

                          id, 0, SEEK_SET);

    ret = qmp_fd(fixture->fd, cmd);

    qmp_assert_no_error(ret);

    val = qdict_get_qdict(ret, "return");

    count = qdict_get_int(val, "position");

    eof = qdict_get_bool(val, "eof");

    g_assert_cmpint(count, ==, 0);

    g_assert(!eof);

    QDECREF(ret);

    g_free(cmd);



    /* read */

    cmd = g_strdup_printf("{'execute': 'guest-file-read',"

                          " 'arguments': { 'handle': %" PRId64 "} }",

                          id);

    ret = qmp_fd(fixture->fd, cmd);

    val = qdict_get_qdict(ret, "return");

    count = qdict_get_int(val, "count");

    eof = qdict_get_bool(val, "eof");

    b64 = qdict_get_str(val, "buf-b64");

    g_assert_cmpint(count, ==, sizeof(helloworld));

    g_assert(eof);

    g_assert_cmpstr(b64, ==, enc);

    QDECREF(ret);

    g_free(cmd);

    g_free(enc);



    /* close */

    cmd = g_strdup_printf("{'execute': 'guest-file-close',"

                          " 'arguments': {'handle': %" PRId64 "} }",

                          id);

    ret = qmp_fd(fixture->fd, cmd);

    QDECREF(ret);

    g_free(cmd);

}
