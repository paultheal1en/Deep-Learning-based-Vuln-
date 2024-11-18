static void img_snapshot(int argc, char **argv)

{

    BlockDriverState *bs;

    QEMUSnapshotInfo sn;

    char *filename, *snapshot_name = NULL;

    int c, ret;

    int action = 0;

    qemu_timeval tv;



    /* Parse commandline parameters */

    for(;;) {

        c = getopt(argc, argv, "la:c:d:h");

        if (c == -1)

            break;

        switch(c) {

        case 'h':

            help();

            return;

        case 'l':

            if (action) {

                help();

                return;

            }

            action = SNAPSHOT_LIST;

            break;

        case 'a':

            if (action) {

                help();

                return;

            }

            action = SNAPSHOT_APPLY;

            snapshot_name = optarg;

            break;

        case 'c':

            if (action) {

                help();

                return;

            }

            action = SNAPSHOT_CREATE;

            snapshot_name = optarg;

            break;

        case 'd':

            if (action) {

                help();

                return;

            }

            action = SNAPSHOT_DELETE;

            snapshot_name = optarg;

            break;

        }

    }



    if (optind >= argc)

        help();

    filename = argv[optind++];



    /* Open the image */

    bs = bdrv_new("");

    if (!bs)

        error("Not enough memory");



    if (bdrv_open2(bs, filename, 0, NULL) < 0) {

        error("Could not open '%s'", filename);

    }



    /* Perform the requested action */

    switch(action) {

    case SNAPSHOT_LIST:

        dump_snapshots(bs);

        break;



    case SNAPSHOT_CREATE:

        memset(&sn, 0, sizeof(sn));

        pstrcpy(sn.name, sizeof(sn.name), snapshot_name);



        qemu_gettimeofday(&tv);

        sn.date_sec = tv.tv_sec;

        sn.date_nsec = tv.tv_usec * 1000;



        ret = bdrv_snapshot_create(bs, &sn);

        if (ret)

            error("Could not create snapshot '%s': %d (%s)",

                snapshot_name, ret, strerror(-ret));

        break;



    case SNAPSHOT_APPLY:

        ret = bdrv_snapshot_goto(bs, snapshot_name);

        if (ret)

            error("Could not apply snapshot '%s': %d (%s)",

                snapshot_name, ret, strerror(-ret));

        break;



    case SNAPSHOT_DELETE:

        ret = bdrv_snapshot_delete(bs, snapshot_name);

        if (ret)

            error("Could not delete snapshot '%s': %d (%s)",

                snapshot_name, ret, strerror(-ret));

        break;

    }



    /* Cleanup */

    bdrv_delete(bs);

}
