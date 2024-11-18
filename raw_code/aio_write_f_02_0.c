static int aio_write_f(BlockBackend *blk, int argc, char **argv)

{

    int nr_iov, c;

    int pattern = 0xcd;

    struct aio_ctx *ctx = g_new0(struct aio_ctx, 1);

    int flags = 0;



    ctx->blk = blk;

    while ((c = getopt(argc, argv, "CfqP:uz")) != -1) {

        switch (c) {

        case 'C':

            ctx->Cflag = true;

            break;

        case 'f':

            flags |= BDRV_REQ_FUA;

            break;

        case 'q':

            ctx->qflag = true;

            break;

        case 'u':

            flags |= BDRV_REQ_MAY_UNMAP;

            break;

        case 'P':

            pattern = parse_pattern(optarg);

            if (pattern < 0) {

                g_free(ctx);

                return 0;

            }

            break;

        case 'z':

            ctx->zflag = true;

            break;

        default:

            g_free(ctx);

            return qemuio_command_usage(&aio_write_cmd);

        }

    }



    if (optind > argc - 2) {

        g_free(ctx);

        return qemuio_command_usage(&aio_write_cmd);

    }



    if (ctx->zflag && optind != argc - 2) {

        printf("-z supports only a single length parameter\n");

        g_free(ctx);

        return 0;

    }



    if ((flags & BDRV_REQ_MAY_UNMAP) && !ctx->zflag) {

        printf("-u requires -z to be specified\n");

        g_free(ctx);

        return 0;

    }



    if (ctx->zflag && ctx->Pflag) {

        printf("-z and -P cannot be specified at the same time\n");

        g_free(ctx);

        return 0;

    }



    ctx->offset = cvtnum(argv[optind]);

    if (ctx->offset < 0) {

        print_cvtnum_err(ctx->offset, argv[optind]);

        g_free(ctx);

        return 0;

    }

    optind++;



    if (ctx->zflag) {

        int64_t count = cvtnum(argv[optind]);

        if (count < 0) {

            print_cvtnum_err(count, argv[optind]);

            g_free(ctx);

            return 0;

        }



        ctx->qiov.size = count;

        blk_aio_write_zeroes(blk, ctx->offset, count, flags, aio_write_done,

                             ctx);

    } else {

        nr_iov = argc - optind;

        ctx->buf = create_iovec(blk, &ctx->qiov, &argv[optind], nr_iov,

                                pattern);

        if (ctx->buf == NULL) {

            block_acct_invalid(blk_get_stats(blk), BLOCK_ACCT_WRITE);

            g_free(ctx);

            return 0;

        }



        gettimeofday(&ctx->t1, NULL);

        block_acct_start(blk_get_stats(blk), &ctx->acct, ctx->qiov.size,

                         BLOCK_ACCT_WRITE);



        blk_aio_pwritev(blk, ctx->offset, &ctx->qiov, flags, aio_write_done,

                        ctx);

    }

    return 0;

}
