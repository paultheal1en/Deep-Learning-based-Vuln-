static void test_qga_config(gconstpointer data)

{

    GError *error = NULL;

    char *cwd, *cmd, *out, *err, *str, **strv, **argv = NULL;

    char *env[2];

    int status;

    gsize n;

    GKeyFile *kf;



    cwd = g_get_current_dir();

    cmd = g_strdup_printf("%s%cqemu-ga -D",

                          cwd, G_DIR_SEPARATOR);


    g_shell_parse_argv(cmd, NULL, &argv, &error);


    g_assert_no_error(error);



    env[0] = g_strdup_printf("QGA_CONF=tests%cdata%ctest-qga-config",

                             G_DIR_SEPARATOR, G_DIR_SEPARATOR);

    env[1] = NULL;

    g_spawn_sync(NULL, argv, env, 0,

                 NULL, NULL, &out, &err, &status, &error);

    g_strfreev(argv);



    g_assert_no_error(error);

    g_assert_cmpstr(err, ==, "");

    g_assert_cmpint(status, ==, 0);



    kf = g_key_file_new();

    g_key_file_load_from_data(kf, out, -1, G_KEY_FILE_NONE, &error);

    g_assert_no_error(error);



    str = g_key_file_get_start_group(kf);

    g_assert_cmpstr(str, ==, "general");

    g_free(str);



    g_assert_false(g_key_file_get_boolean(kf, "general", "daemon", &error));

    g_assert_no_error(error);



    str = g_key_file_get_string(kf, "general", "method", &error);

    g_assert_no_error(error);

    g_assert_cmpstr(str, ==, "virtio-serial");

    g_free(str);



    str = g_key_file_get_string(kf, "general", "path", &error);

    g_assert_no_error(error);

    g_assert_cmpstr(str, ==, "/path/to/org.qemu.guest_agent.0");

    g_free(str);



    str = g_key_file_get_string(kf, "general", "pidfile", &error);

    g_assert_no_error(error);

    g_assert_cmpstr(str, ==, "/var/foo/qemu-ga.pid");

    g_free(str);



    str = g_key_file_get_string(kf, "general", "statedir", &error);

    g_assert_no_error(error);

    g_assert_cmpstr(str, ==, "/var/state");

    g_free(str);



    g_assert_true(g_key_file_get_boolean(kf, "general", "verbose", &error));

    g_assert_no_error(error);



    strv = g_key_file_get_string_list(kf, "general", "blacklist", &n, &error);

    g_assert_cmpint(n, ==, 2);

#if GLIB_CHECK_VERSION(2, 44, 0)

    g_assert_true(g_strv_contains((const char * const *)strv,

                                  "guest-ping"));

    g_assert_true(g_strv_contains((const char * const *)strv,

                                  "guest-get-time"));

#endif

    g_assert_no_error(error);

    g_strfreev(strv);



    g_free(out);

    g_free(err);

    g_free(env[0]);

    g_key_file_free(kf);

}