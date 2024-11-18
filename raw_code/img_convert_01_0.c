static int img_convert(int argc, char **argv)

{

    int c, n, n1, bs_n, bs_i, compress, cluster_sectors, skip_create;

    int64_t ret = 0;

    int progress = 0, flags;

    const char *fmt, *out_fmt, *cache, *out_baseimg, *out_filename;

    BlockDriver *drv, *proto_drv;

    BlockDriverState **bs = NULL, *out_bs = NULL;

    int64_t total_sectors, nb_sectors, sector_num, bs_offset;

    int64_t *bs_sectors = NULL;

    uint8_t * buf = NULL;

    size_t bufsectors = IO_BUF_SIZE / BDRV_SECTOR_SIZE;

    const uint8_t *buf1;

    BlockDriverInfo bdi;

    QemuOpts *opts = NULL;

    QemuOptsList *create_opts = NULL;

    const char *out_baseimg_param;

    char *options = NULL;

    const char *snapshot_name = NULL;

    int min_sparse = 8; /* Need at least 4k of zeros for sparse detection */

    bool quiet = false;

    Error *local_err = NULL;

    QemuOpts *sn_opts = NULL;



    fmt = NULL;

    out_fmt = "raw";

    cache = "unsafe";

    out_baseimg = NULL;

    compress = 0;

    skip_create = 0;

    for(;;) {

        c = getopt(argc, argv, "f:O:B:s:hce6o:pS:t:qnl:");

        if (c == -1) {

            break;

        }

        switch(c) {

        case '?':

        case 'h':

            help();

            break;

        case 'f':

            fmt = optarg;

            break;

        case 'O':

            out_fmt = optarg;

            break;

        case 'B':

            out_baseimg = optarg;

            break;

        case 'c':

            compress = 1;

            break;

        case 'e':

            error_report("option -e is deprecated, please use \'-o "

                  "encryption\' instead!");

            ret = -1;

            goto fail_getopt;

        case '6':

            error_report("option -6 is deprecated, please use \'-o "

                  "compat6\' instead!");

            ret = -1;

            goto fail_getopt;

        case 'o':

            if (!is_valid_option_list(optarg)) {

                error_report("Invalid option list: %s", optarg);

                ret = -1;

                goto fail_getopt;

            }

            if (!options) {

                options = g_strdup(optarg);

            } else {

                char *old_options = options;

                options = g_strdup_printf("%s,%s", options, optarg);

                g_free(old_options);

            }

            break;

        case 's':

            snapshot_name = optarg;

            break;

        case 'l':

            if (strstart(optarg, SNAPSHOT_OPT_BASE, NULL)) {

                sn_opts = qemu_opts_parse(&internal_snapshot_opts, optarg, 0);

                if (!sn_opts) {

                    error_report("Failed in parsing snapshot param '%s'",

                                 optarg);

                    ret = -1;

                    goto fail_getopt;

                }

            } else {

                snapshot_name = optarg;

            }

            break;

        case 'S':

        {

            int64_t sval;

            char *end;

            sval = strtosz_suffix(optarg, &end, STRTOSZ_DEFSUFFIX_B);

            if (sval < 0 || *end) {

                error_report("Invalid minimum zero buffer size for sparse output specified");

                ret = -1;

                goto fail_getopt;

            }



            min_sparse = sval / BDRV_SECTOR_SIZE;

            break;

        }

        case 'p':

            progress = 1;

            break;

        case 't':

            cache = optarg;

            break;

        case 'q':

            quiet = true;

            break;

        case 'n':

            skip_create = 1;

            break;

        }

    }



    /* Initialize before goto out */

    if (quiet) {

        progress = 0;

    }

    qemu_progress_init(progress, 1.0);





    bs_n = argc - optind - 1;

    out_filename = bs_n >= 1 ? argv[argc - 1] : NULL;



    if (options && has_help_option(options)) {

        ret = print_block_option_help(out_filename, out_fmt);

        goto out;

    }



    if (bs_n < 1) {

        error_exit("Must specify image file name");

    }





    if (bs_n > 1 && out_baseimg) {

        error_report("-B makes no sense when concatenating multiple input "

                     "images");

        ret = -1;

        goto out;

    }



    qemu_progress_print(0, 100);



    bs = g_new0(BlockDriverState *, bs_n);

    bs_sectors = g_new(int64_t, bs_n);



    total_sectors = 0;

    for (bs_i = 0; bs_i < bs_n; bs_i++) {

        char *id = bs_n > 1 ? g_strdup_printf("source %d", bs_i)

                            : g_strdup("source");

        bs[bs_i] = bdrv_new_open(id, argv[optind + bs_i], fmt, BDRV_O_FLAGS,

                                 true, quiet);

        g_free(id);

        if (!bs[bs_i]) {

            error_report("Could not open '%s'", argv[optind + bs_i]);

            ret = -1;

            goto out;

        }

        bs_sectors[bs_i] = bdrv_nb_sectors(bs[bs_i]);

        if (bs_sectors[bs_i] < 0) {

            error_report("Could not get size of %s: %s",

                         argv[optind + bs_i], strerror(-bs_sectors[bs_i]));

            ret = -1;

            goto out;

        }

        total_sectors += bs_sectors[bs_i];

    }



    if (sn_opts) {

        ret = bdrv_snapshot_load_tmp(bs[0],

                                     qemu_opt_get(sn_opts, SNAPSHOT_OPT_ID),

                                     qemu_opt_get(sn_opts, SNAPSHOT_OPT_NAME),

                                     &local_err);

    } else if (snapshot_name != NULL) {

        if (bs_n > 1) {

            error_report("No support for concatenating multiple snapshot");

            ret = -1;

            goto out;

        }



        bdrv_snapshot_load_tmp_by_id_or_name(bs[0], snapshot_name, &local_err);

    }

    if (local_err) {

        error_report("Failed to load snapshot: %s",

                     error_get_pretty(local_err));

        error_free(local_err);

        ret = -1;

        goto out;

    }



    /* Find driver and parse its options */

    drv = bdrv_find_format(out_fmt);

    if (!drv) {

        error_report("Unknown file format '%s'", out_fmt);

        ret = -1;

        goto out;

    }



    proto_drv = bdrv_find_protocol(out_filename, true);

    if (!proto_drv) {

        error_report("Unknown protocol '%s'", out_filename);

        ret = -1;

        goto out;

    }



    create_opts = qemu_opts_append(create_opts, drv->create_opts);

    create_opts = qemu_opts_append(create_opts, proto_drv->create_opts);



    opts = qemu_opts_create(create_opts, NULL, 0, &error_abort);

    if (options && qemu_opts_do_parse(opts, options, NULL)) {

        error_report("Invalid options for file format '%s'", out_fmt);

        ret = -1;

        goto out;

    }



    qemu_opt_set_number(opts, BLOCK_OPT_SIZE, total_sectors * 512);

    ret = add_old_style_options(out_fmt, opts, out_baseimg, NULL);

    if (ret < 0) {

        goto out;

    }



    /* Get backing file name if -o backing_file was used */

    out_baseimg_param = qemu_opt_get(opts, BLOCK_OPT_BACKING_FILE);

    if (out_baseimg_param) {

        out_baseimg = out_baseimg_param;

    }



    /* Check if compression is supported */

    if (compress) {

        bool encryption =

            qemu_opt_get_bool(opts, BLOCK_OPT_ENCRYPT, false);

        const char *preallocation =

            qemu_opt_get(opts, BLOCK_OPT_PREALLOC);



        if (!drv->bdrv_write_compressed) {

            error_report("Compression not supported for this file format");

            ret = -1;

            goto out;

        }



        if (encryption) {

            error_report("Compression and encryption not supported at "

                         "the same time");

            ret = -1;

            goto out;

        }



        if (preallocation

            && strcmp(preallocation, "off"))

        {

            error_report("Compression and preallocation not supported at "

                         "the same time");

            ret = -1;

            goto out;

        }

    }



    if (!skip_create) {

        /* Create the new image */

        ret = bdrv_create(drv, out_filename, opts, &local_err);

        if (ret < 0) {

            error_report("%s: error while converting %s: %s",

                         out_filename, out_fmt, error_get_pretty(local_err));

            error_free(local_err);

            goto out;

        }

    }



    flags = min_sparse ? (BDRV_O_RDWR | BDRV_O_UNMAP) : BDRV_O_RDWR;

    ret = bdrv_parse_cache_flags(cache, &flags);

    if (ret < 0) {

        error_report("Invalid cache option: %s", cache);

        goto out;

    }



    out_bs = bdrv_new_open("target", out_filename, out_fmt, flags, true, quiet);

    if (!out_bs) {

        ret = -1;

        goto out;

    }



    bs_i = 0;

    bs_offset = 0;



    /* increase bufsectors from the default 4096 (2M) if opt_transfer_length

     * or discard_alignment of the out_bs is greater. Limit to 32768 (16MB)

     * as maximum. */

    bufsectors = MIN(32768,

                     MAX(bufsectors, MAX(out_bs->bl.opt_transfer_length,

                                         out_bs->bl.discard_alignment))

                    );



    buf = qemu_blockalign(out_bs, bufsectors * BDRV_SECTOR_SIZE);



    if (skip_create) {

        int64_t output_sectors = bdrv_nb_sectors(out_bs);

        if (output_sectors < 0) {

            error_report("unable to get output image length: %s\n",

                         strerror(-output_sectors));

            ret = -1;

            goto out;

        } else if (output_sectors < total_sectors) {

            error_report("output file is smaller than input file");

            ret = -1;

            goto out;

        }

    }



    cluster_sectors = 0;

    ret = bdrv_get_info(out_bs, &bdi);

    if (ret < 0) {

        if (compress) {

            error_report("could not get block driver info");

            goto out;

        }

    } else {

        compress = compress || bdi.needs_compressed_writes;

        cluster_sectors = bdi.cluster_size / BDRV_SECTOR_SIZE;

    }



    if (compress) {

        if (cluster_sectors <= 0 || cluster_sectors > bufsectors) {

            error_report("invalid cluster size");

            ret = -1;

            goto out;

        }

        sector_num = 0;



        nb_sectors = total_sectors;



        for(;;) {

            int64_t bs_num;

            int remainder;

            uint8_t *buf2;



            nb_sectors = total_sectors - sector_num;

            if (nb_sectors <= 0)

                break;

            if (nb_sectors >= cluster_sectors)

                n = cluster_sectors;

            else

                n = nb_sectors;



            bs_num = sector_num - bs_offset;

            assert (bs_num >= 0);

            remainder = n;

            buf2 = buf;

            while (remainder > 0) {

                int nlow;

                while (bs_num == bs_sectors[bs_i]) {

                    bs_offset += bs_sectors[bs_i];

                    bs_i++;

                    assert (bs_i < bs_n);

                    bs_num = 0;

                    /* printf("changing part: sector_num=%" PRId64 ", "

                       "bs_i=%d, bs_offset=%" PRId64 ", bs_sectors=%" PRId64

                       "\n", sector_num, bs_i, bs_offset, bs_sectors[bs_i]); */

                }

                assert (bs_num < bs_sectors[bs_i]);



                nlow = remainder > bs_sectors[bs_i] - bs_num

                    ? bs_sectors[bs_i] - bs_num : remainder;



                ret = bdrv_read(bs[bs_i], bs_num, buf2, nlow);

                if (ret < 0) {

                    error_report("error while reading sector %" PRId64 ": %s",

                                 bs_num, strerror(-ret));

                    goto out;

                }



                buf2 += nlow * 512;

                bs_num += nlow;



                remainder -= nlow;

            }

            assert (remainder == 0);



            if (!buffer_is_zero(buf, n * BDRV_SECTOR_SIZE)) {

                ret = bdrv_write_compressed(out_bs, sector_num, buf, n);

                if (ret != 0) {

                    error_report("error while compressing sector %" PRId64

                                 ": %s", sector_num, strerror(-ret));

                    goto out;

                }

            }

            sector_num += n;

            qemu_progress_print(100.0 * sector_num / total_sectors, 0);

        }

        /* signal EOF to align */

        bdrv_write_compressed(out_bs, 0, NULL, 0);

    } else {

        int64_t sectors_to_read, sectors_read, sector_num_next_status;

        bool count_allocated_sectors;

        int has_zero_init = min_sparse ? bdrv_has_zero_init(out_bs) : 0;



        if (!has_zero_init && bdrv_can_write_zeroes_with_unmap(out_bs)) {

            ret = bdrv_make_zero(out_bs, BDRV_REQ_MAY_UNMAP);

            if (ret < 0) {

                goto out;

            }

            has_zero_init = 1;

        }



        sectors_to_read = total_sectors;

        count_allocated_sectors = progress && (out_baseimg || has_zero_init);

restart:

        sector_num = 0; // total number of sectors converted so far

        sectors_read = 0;

        sector_num_next_status = 0;



        for(;;) {

            nb_sectors = total_sectors - sector_num;

            if (nb_sectors <= 0) {

                if (count_allocated_sectors) {

                    sectors_to_read = sectors_read;

                    count_allocated_sectors = false;

                    goto restart;

                }

                ret = 0;

                break;

            }



            while (sector_num - bs_offset >= bs_sectors[bs_i]) {

                bs_offset += bs_sectors[bs_i];

                bs_i ++;

                assert (bs_i < bs_n);

                /* printf("changing part: sector_num=%" PRId64 ", bs_i=%d, "

                  "bs_offset=%" PRId64 ", bs_sectors=%" PRId64 "\n",

                   sector_num, bs_i, bs_offset, bs_sectors[bs_i]); */

            }



            if ((out_baseimg || has_zero_init) &&

                sector_num >= sector_num_next_status) {

                n = nb_sectors > INT_MAX ? INT_MAX : nb_sectors;

                ret = bdrv_get_block_status(bs[bs_i], sector_num - bs_offset,

                                            n, &n1);

                if (ret < 0) {

                    error_report("error while reading block status of sector %"

                                 PRId64 ": %s", sector_num - bs_offset,

                                 strerror(-ret));

                    goto out;

                }

                /* If the output image is zero initialized, we are not working

                 * on a shared base and the input is zero we can skip the next

                 * n1 sectors */

                if (has_zero_init && !out_baseimg && (ret & BDRV_BLOCK_ZERO)) {

                    sector_num += n1;

                    continue;

                }

                /* If the output image is being created as a copy on write

                 * image, assume that sectors which are unallocated in the

                 * input image are present in both the output's and input's

                 * base images (no need to copy them). */

                if (out_baseimg) {

                    if (!(ret & BDRV_BLOCK_DATA)) {

                        sector_num += n1;

                        continue;

                    }

                    /* The next 'n1' sectors are allocated in the input image.

                     * Copy only those as they may be followed by unallocated

                     * sectors. */

                    nb_sectors = n1;

                }

                /* avoid redundant callouts to get_block_status */

                sector_num_next_status = sector_num + n1;

            }



            n = MIN(nb_sectors, bufsectors);



            /* round down request length to an aligned sector, but

             * do not bother doing this on short requests. They happen

             * when we found an all-zero area, and the next sector to

             * write will not be sector_num + n. */

            if (cluster_sectors > 0 && n >= cluster_sectors) {

                int64_t next_aligned_sector = (sector_num + n);

                next_aligned_sector -= next_aligned_sector % cluster_sectors;

                if (sector_num + n > next_aligned_sector) {

                    n = next_aligned_sector - sector_num;

                }

            }



            n = MIN(n, bs_sectors[bs_i] - (sector_num - bs_offset));



            sectors_read += n;

            if (count_allocated_sectors) {

                sector_num += n;

                continue;

            }



            n1 = n;

            ret = bdrv_read(bs[bs_i], sector_num - bs_offset, buf, n);

            if (ret < 0) {

                error_report("error while reading sector %" PRId64 ": %s",

                             sector_num - bs_offset, strerror(-ret));

                goto out;

            }

            /* NOTE: at the same time we convert, we do not write zero

               sectors to have a chance to compress the image. Ideally, we

               should add a specific call to have the info to go faster */

            buf1 = buf;

            while (n > 0) {

                if (!has_zero_init ||

                    is_allocated_sectors_min(buf1, n, &n1, min_sparse)) {

                    ret = bdrv_write(out_bs, sector_num, buf1, n1);

                    if (ret < 0) {

                        error_report("error while writing sector %" PRId64

                                     ": %s", sector_num, strerror(-ret));

                        goto out;

                    }

                }

                sector_num += n1;

                n -= n1;

                buf1 += n1 * 512;

            }

            qemu_progress_print(100.0 * sectors_read / sectors_to_read, 0);

        }

    }

out:

    if (!ret) {

        qemu_progress_print(100, 0);

    }

    qemu_progress_end();

    qemu_opts_del(opts);

    qemu_opts_free(create_opts);

    qemu_vfree(buf);

    if (sn_opts) {

        qemu_opts_del(sn_opts);

    }

    if (out_bs) {

        bdrv_unref(out_bs);

    }

    if (bs) {

        for (bs_i = 0; bs_i < bs_n; bs_i++) {

            if (bs[bs_i]) {

                bdrv_unref(bs[bs_i]);

            }

        }

        g_free(bs);

    }

    g_free(bs_sectors);

fail_getopt:

    g_free(options);



    if (ret) {

        return 1;

    }

    return 0;

}
