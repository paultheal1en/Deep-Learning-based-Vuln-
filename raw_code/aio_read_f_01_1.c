static int aio_read_f(BlockBackend *blk, int argc, char **argv)

{

    int nr_iov, c;

    struct aio_ctx *ctx = g_new0(struct aio_ctx, 1);



    ctx->blk = blk;

    while ((c = getopt(argc, argv, "CP:qv")) != -1) {

        switch (c) {

        case 'C':

            ctx->Cflag = 1;

            break;

        case 'P':

            ctx->Pflag = 1;

            ctx->pattern = parse_pattern(optarg);

            if (ctx->pattern < 0) {

                g_free(ctx);

                return 0;

            }

            break;

        case 'q':

            ctx->qflag = 1;

            break;

        case 'v':

            ctx->vflag = 1;

            break;

        default:

            g_free(ctx);

            return qemuio_command_usage(&aio_read_cmd);

        }

    }



    if (optind > argc - 2) {

        g_free(ctx);

        return qemuio_command_usage(&aio_read_cmd);

    }



    ctx->offset = cvtnum(argv[optind]);

    if (ctx->offset < 0) {

        print_cvtnum_err(ctx->offset, argv[optind]);

        g_free(ctx);

        return 0;

    }

    optind++;



    if (ctx->offset & 0x1ff) {

        printf("offset %" PRId64 " is not sector aligned\n",

               ctx->offset);


        g_free(ctx);

        return 0;

    }



    nr_iov = argc - optind;

    ctx->buf = create_iovec(blk, &ctx->qiov, &argv[optind], nr_iov, 0xab);

    if (ctx->buf == NULL) {


        g_free(ctx);

        return 0;

    }



    gettimeofday(&ctx->t1, NULL);

    block_acct_start(blk_get_stats(blk), &ctx->acct, ctx->qiov.size,

                     BLOCK_ACCT_READ);

    blk_aio_readv(blk, ctx->offset >> 9, &ctx->qiov,

                  ctx->qiov.size >> 9, aio_read_done, ctx);

    return 0;

}