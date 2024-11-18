static void test_ide_drive_user(const char *dev, bool trans)

{

    char *argv[256], *opts;

    int argc;

    int secs = img_secs[backend_small];

    const CHST expected_chst = { secs / (4 * 32) , 4, 32, trans };



    argc = setup_common(argv, ARRAY_SIZE(argv));

    opts = g_strdup_printf("%s,%s%scyls=%d,heads=%d,secs=%d",

                           dev ?: "",

                           trans && dev ? "bios-chs-" : "",

                           trans ? "trans=lba," : "",

                           expected_chst.cyls, expected_chst.heads,

                           expected_chst.secs);

    cur_ide[0] = &expected_chst;

    argc = setup_ide(argc, argv, ARRAY_SIZE(argv),

                     0, dev ? opts : NULL, backend_small, mbr_chs,

                     dev ? "" : opts);

    g_free(opts);

    qtest_start(g_strjoinv(" ", argv));

    test_cmos();

    qtest_end();

}
