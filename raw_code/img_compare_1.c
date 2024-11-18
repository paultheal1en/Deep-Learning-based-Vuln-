static int img_compare(int argc, char **argv)

{

    const char *fmt1 = NULL, *fmt2 = NULL, *cache, *filename1, *filename2;

    BlockBackend *blk1, *blk2;

    BlockDriverState *bs1, *bs2;

    int64_t total_sectors1, total_sectors2;

    uint8_t *buf1 = NULL, *buf2 = NULL;

    int pnum1, pnum2;

    int allocated1, allocated2;

    int ret = 0; /* return value - 0 Ident, 1 Different, >1 Error */

    bool progress = false, quiet = false, strict = false;

    int flags;

    bool writethrough;

    int64_t total_sectors;

    int64_t sector_num = 0;

    int64_t nb_sectors;

    int c, pnum;

    uint64_t progress_base;

    bool image_opts = false;



    cache = BDRV_DEFAULT_CACHE;

    for (;;) {

        static const struct option long_options[] = {

            {"help", no_argument, 0, 'h'},

            {"object", required_argument, 0, OPTION_OBJECT},

            {"image-opts", no_argument, 0, OPTION_IMAGE_OPTS},

            {0, 0, 0, 0}

        };

        c = getopt_long(argc, argv, "hf:F:T:pqs",

                        long_options, NULL);

        if (c == -1) {

            break;

        }

        switch (c) {

        case '?':

        case 'h':

            help();

            break;

        case 'f':

            fmt1 = optarg;

            break;

        case 'F':

            fmt2 = optarg;

            break;

        case 'T':

            cache = optarg;

            break;

        case 'p':

            progress = true;

            break;

        case 'q':

            quiet = true;

            break;

        case 's':

            strict = true;

            break;

        case OPTION_OBJECT: {

            QemuOpts *opts;

            opts = qemu_opts_parse_noisily(&qemu_object_opts,

                                           optarg, true);

            if (!opts) {

                ret = 2;

                goto out4;

            }

        }   break;

        case OPTION_IMAGE_OPTS:

            image_opts = true;

            break;

        }

    }



    /* Progress is not shown in Quiet mode */

    if (quiet) {

        progress = false;

    }





    if (optind != argc - 2) {

        error_exit("Expecting two image file names");

    }

    filename1 = argv[optind++];

    filename2 = argv[optind++];



    if (qemu_opts_foreach(&qemu_object_opts,

                          user_creatable_add_opts_foreach,

                          NULL, NULL)) {

        ret = 2;

        goto out4;

    }



    /* Initialize before goto out */

    qemu_progress_init(progress, 2.0);



    flags = 0;

    ret = bdrv_parse_cache_mode(cache, &flags, &writethrough);

    if (ret < 0) {

        error_report("Invalid source cache option: %s", cache);

        ret = 2;

        goto out3;

    }



    blk1 = img_open(image_opts, filename1, fmt1, flags, writethrough, quiet);

    if (!blk1) {

        ret = 2;

        goto out3;

    }



    blk2 = img_open(image_opts, filename2, fmt2, flags, writethrough, quiet);

    if (!blk2) {

        ret = 2;

        goto out2;

    }

    bs1 = blk_bs(blk1);

    bs2 = blk_bs(blk2);



    buf1 = blk_blockalign(blk1, IO_BUF_SIZE);

    buf2 = blk_blockalign(blk2, IO_BUF_SIZE);

    total_sectors1 = blk_nb_sectors(blk1);

    if (total_sectors1 < 0) {

        error_report("Can't get size of %s: %s",

                     filename1, strerror(-total_sectors1));

        ret = 4;

        goto out;

    }

    total_sectors2 = blk_nb_sectors(blk2);

    if (total_sectors2 < 0) {

        error_report("Can't get size of %s: %s",

                     filename2, strerror(-total_sectors2));

        ret = 4;

        goto out;

    }

    total_sectors = MIN(total_sectors1, total_sectors2);

    progress_base = MAX(total_sectors1, total_sectors2);



    qemu_progress_print(0, 100);



    if (strict && total_sectors1 != total_sectors2) {

        ret = 1;

        qprintf(quiet, "Strict mode: Image size mismatch!\n");

        goto out;

    }



    for (;;) {

        int64_t status1, status2;

        BlockDriverState *file;



        nb_sectors = sectors_to_process(total_sectors, sector_num);

        if (nb_sectors <= 0) {

            break;

        }

        status1 = bdrv_get_block_status_above(bs1, NULL, sector_num,

                                              total_sectors1 - sector_num,

                                              &pnum1, &file);

        if (status1 < 0) {

            ret = 3;

            error_report("Sector allocation test failed for %s", filename1);

            goto out;

        }

        allocated1 = status1 & BDRV_BLOCK_ALLOCATED;



        status2 = bdrv_get_block_status_above(bs2, NULL, sector_num,

                                              total_sectors2 - sector_num,

                                              &pnum2, &file);

        if (status2 < 0) {

            ret = 3;

            error_report("Sector allocation test failed for %s", filename2);

            goto out;

        }

        allocated2 = status2 & BDRV_BLOCK_ALLOCATED;

        if (pnum1) {

            nb_sectors = MIN(nb_sectors, pnum1);

        }

        if (pnum2) {

            nb_sectors = MIN(nb_sectors, pnum2);

        }



        if (strict) {

            if ((status1 & ~BDRV_BLOCK_OFFSET_MASK) !=

                (status2 & ~BDRV_BLOCK_OFFSET_MASK)) {

                ret = 1;

                qprintf(quiet, "Strict mode: Offset %" PRId64

                        " block status mismatch!\n",

                        sectors_to_bytes(sector_num));

                goto out;

            }

        }

        if ((status1 & BDRV_BLOCK_ZERO) && (status2 & BDRV_BLOCK_ZERO)) {

            nb_sectors = MIN(pnum1, pnum2);

        } else if (allocated1 == allocated2) {

            if (allocated1) {

                ret = blk_pread(blk1, sector_num << BDRV_SECTOR_BITS, buf1,

                                nb_sectors << BDRV_SECTOR_BITS);

                if (ret < 0) {

                    error_report("Error while reading offset %" PRId64 " of %s:"

                                 " %s", sectors_to_bytes(sector_num), filename1,

                                 strerror(-ret));

                    ret = 4;

                    goto out;

                }

                ret = blk_pread(blk2, sector_num << BDRV_SECTOR_BITS, buf2,

                                nb_sectors << BDRV_SECTOR_BITS);

                if (ret < 0) {

                    error_report("Error while reading offset %" PRId64

                                 " of %s: %s", sectors_to_bytes(sector_num),

                                 filename2, strerror(-ret));

                    ret = 4;

                    goto out;

                }

                ret = compare_sectors(buf1, buf2, nb_sectors, &pnum);

                if (ret || pnum != nb_sectors) {

                    qprintf(quiet, "Content mismatch at offset %" PRId64 "!\n",

                            sectors_to_bytes(

                                ret ? sector_num : sector_num + pnum));

                    ret = 1;

                    goto out;

                }

            }

        } else {



            if (allocated1) {

                ret = check_empty_sectors(blk1, sector_num, nb_sectors,

                                          filename1, buf1, quiet);

            } else {

                ret = check_empty_sectors(blk2, sector_num, nb_sectors,

                                          filename2, buf1, quiet);

            }

            if (ret) {

                if (ret < 0) {

                    error_report("Error while reading offset %" PRId64 ": %s",

                                 sectors_to_bytes(sector_num), strerror(-ret));

                    ret = 4;

                }

                goto out;

            }

        }

        sector_num += nb_sectors;

        qemu_progress_print(((float) nb_sectors / progress_base)*100, 100);

    }



    if (total_sectors1 != total_sectors2) {

        BlockBackend *blk_over;

        int64_t total_sectors_over;

        const char *filename_over;



        qprintf(quiet, "Warning: Image size mismatch!\n");

        if (total_sectors1 > total_sectors2) {

            total_sectors_over = total_sectors1;

            blk_over = blk1;

            filename_over = filename1;

        } else {

            total_sectors_over = total_sectors2;

            blk_over = blk2;

            filename_over = filename2;

        }



        for (;;) {

            nb_sectors = sectors_to_process(total_sectors_over, sector_num);

            if (nb_sectors <= 0) {

                break;

            }

            ret = bdrv_is_allocated_above(blk_bs(blk_over), NULL, sector_num,

                                          nb_sectors, &pnum);

            if (ret < 0) {

                ret = 3;

                error_report("Sector allocation test failed for %s",

                             filename_over);

                goto out;



            }

            nb_sectors = pnum;

            if (ret) {

                ret = check_empty_sectors(blk_over, sector_num, nb_sectors,

                                          filename_over, buf1, quiet);

                if (ret) {

                    if (ret < 0) {

                        error_report("Error while reading offset %" PRId64

                                     " of %s: %s", sectors_to_bytes(sector_num),

                                     filename_over, strerror(-ret));

                        ret = 4;

                    }

                    goto out;

                }

            }

            sector_num += nb_sectors;

            qemu_progress_print(((float) nb_sectors / progress_base)*100, 100);

        }

    }



    qprintf(quiet, "Images are identical.\n");

    ret = 0;



out:

    qemu_vfree(buf1);

    qemu_vfree(buf2);

    blk_unref(blk2);

out2:

    blk_unref(blk1);

out3:

    qemu_progress_end();

out4:

    return ret;

}
