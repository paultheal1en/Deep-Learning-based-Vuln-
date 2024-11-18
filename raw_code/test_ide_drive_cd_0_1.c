static void test_ide_drive_cd_0(void)

{

    char *argv[256];

    int argc, ide_idx;

    Backend i;



    argc = setup_common(argv, ARRAY_SIZE(argv));

    for (i = 0; i <= backend_empty; i++) {

        ide_idx = backend_empty - i;

        cur_ide[ide_idx] = &hd_chst[i][mbr_blank];

        argc = setup_ide(argc, argv, ARRAY_SIZE(argv),

                         ide_idx, NULL, i, mbr_blank, "");

    }

    qtest_start(g_strjoinv(" ", argv));

    test_cmos();

    qtest_end();

}
