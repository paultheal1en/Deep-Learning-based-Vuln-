static int img_rebase(int argc, char **argv)

{

    BlockDriverState *bs, *bs_old_backing, *bs_new_backing;

    BlockDriver *old_backing_drv, *new_backing_drv;

    char *filename;

    const char *fmt, *out_basefmt, *out_baseimg;

    int c, flags, ret;

    int unsafe = 0;



    /* Parse commandline parameters */

    fmt = NULL;

    out_baseimg = NULL;

    out_basefmt = NULL;



    for(;;) {

        c = getopt(argc, argv, "uhf:F:b:");

        if (c == -1)

            break;

        switch(c) {

        case 'h':

            help();

            return 0;

        case 'f':

            fmt = optarg;

            break;

        case 'F':

            out_basefmt = optarg;

            break;

        case 'b':

            out_baseimg = optarg;

            break;

        case 'u':

            unsafe = 1;

            break;

        }

    }



    if ((optind >= argc) || !out_baseimg)

        help();

    filename = argv[optind++];



    /*

     * Open the images.

     *

     * Ignore the old backing file for unsafe rebase in case we want to correct

     * the reference to a renamed or moved backing file.

     */

    flags = BDRV_O_FLAGS | BDRV_O_RDWR | (unsafe ? BDRV_O_NO_BACKING : 0);

    bs = bdrv_new_open(filename, fmt, flags);



    /* Find the right drivers for the backing files */

    old_backing_drv = NULL;

    new_backing_drv = NULL;



    if (!unsafe && bs->backing_format[0] != '\0') {

        old_backing_drv = bdrv_find_format(bs->backing_format);

        if (old_backing_drv == NULL) {

            error("Invalid format name: '%s'", bs->backing_format);

        }

    }



    if (out_basefmt != NULL) {

        new_backing_drv = bdrv_find_format(out_basefmt);

        if (new_backing_drv == NULL) {

            error("Invalid format name: '%s'", out_basefmt);

        }

    }



    /* For safe rebasing we need to compare old and new backing file */

    if (unsafe) {

        /* Make the compiler happy */

        bs_old_backing = NULL;

        bs_new_backing = NULL;

    } else {

        char backing_name[1024];



        bs_old_backing = bdrv_new("old_backing");

        bdrv_get_backing_filename(bs, backing_name, sizeof(backing_name));

        if (bdrv_open(bs_old_backing, backing_name, BDRV_O_FLAGS,

            old_backing_drv))

        {

            error("Could not open old backing file '%s'", backing_name);

            return -1;

        }



        bs_new_backing = bdrv_new("new_backing");

        if (bdrv_open(bs_new_backing, out_baseimg, BDRV_O_FLAGS | BDRV_O_RDWR,

            new_backing_drv))

        {

            error("Could not open new backing file '%s'", out_baseimg);

            return -1;

        }

    }



    /*

     * Check each unallocated cluster in the COW file. If it is unallocated,

     * accesses go to the backing file. We must therefore compare this cluster

     * in the old and new backing file, and if they differ we need to copy it

     * from the old backing file into the COW file.

     *

     * If qemu-img crashes during this step, no harm is done. The content of

     * the image is the same as the original one at any time.

     */

    if (!unsafe) {

        uint64_t num_sectors;

        uint64_t sector;

        int n, n1;

        uint8_t * buf_old;

        uint8_t * buf_new;



        buf_old = qemu_malloc(IO_BUF_SIZE);

        buf_new = qemu_malloc(IO_BUF_SIZE);



        bdrv_get_geometry(bs, &num_sectors);



        for (sector = 0; sector < num_sectors; sector += n) {



            /* How many sectors can we handle with the next read? */

            if (sector + (IO_BUF_SIZE / 512) <= num_sectors) {

                n = (IO_BUF_SIZE / 512);

            } else {

                n = num_sectors - sector;

            }



            /* If the cluster is allocated, we don't need to take action */

            if (bdrv_is_allocated(bs, sector, n, &n1)) {

                n = n1;

                continue;

            }



            /* Read old and new backing file */

            if (bdrv_read(bs_old_backing, sector, buf_old, n) < 0) {

                error("error while reading from old backing file");

            }

            if (bdrv_read(bs_new_backing, sector, buf_new, n) < 0) {

                error("error while reading from new backing file");

            }



            /* If they differ, we need to write to the COW file */

            uint64_t written = 0;



            while (written < n) {

                int pnum;



                if (compare_sectors(buf_old + written * 512,

                    buf_new + written * 512, n - written, &pnum))

                {

                    ret = bdrv_write(bs, sector + written,

                        buf_old + written * 512, pnum);

                    if (ret < 0) {

                        error("Error while writing to COW image: %s",

                            strerror(-ret));

                    }

                }



                written += pnum;

            }

        }



        qemu_free(buf_old);

        qemu_free(buf_new);

    }



    /*

     * Change the backing file. All clusters that are different from the old

     * backing file are overwritten in the COW file now, so the visible content

     * doesn't change when we switch the backing file.

     */

    ret = bdrv_change_backing_file(bs, out_baseimg, out_basefmt);

    if (ret == -ENOSPC) {

        error("Could not change the backing file to '%s': No space left in "

            "the file header", out_baseimg);

    } else if (ret < 0) {

        error("Could not change the backing file to '%s': %s",

            out_baseimg, strerror(-ret));

    }



    /*

     * TODO At this point it is possible to check if any clusters that are

     * allocated in the COW file are the same in the backing file. If so, they

     * could be dropped from the COW file. Don't do this before switching the

     * backing file, in case of a crash this would lead to corruption.

     */



    /* Cleanup */

    if (!unsafe) {

        bdrv_delete(bs_old_backing);

        bdrv_delete(bs_new_backing);

    }



    bdrv_delete(bs);



    return 0;

}
