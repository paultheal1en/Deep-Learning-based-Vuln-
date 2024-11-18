static void aio_read_done(void *opaque, int ret)

{

    struct aio_ctx *ctx = opaque;

    struct timeval t2;



    gettimeofday(&t2, NULL);



    if (ret < 0) {

        printf("readv failed: %s\n", strerror(-ret));

        goto out;

    }



    if (ctx->Pflag) {

        void *cmp_buf = g_malloc(ctx->qiov.size);



        memset(cmp_buf, ctx->pattern, ctx->qiov.size);

        if (memcmp(ctx->buf, cmp_buf, ctx->qiov.size)) {

            printf("Pattern verification failed at offset %"

                   PRId64 ", %zd bytes\n", ctx->offset, ctx->qiov.size);

        }

        g_free(cmp_buf);

    }



    if (ctx->qflag) {

        goto out;

    }



    if (ctx->vflag) {

        dump_buffer(ctx->buf, ctx->offset, ctx->qiov.size);

    }



    /* Finally, report back -- -C gives a parsable format */

    t2 = tsub(t2, ctx->t1);

    print_report("read", &t2, ctx->offset, ctx->qiov.size,

                 ctx->qiov.size, 1, ctx->Cflag);

out:

    qemu_io_free(ctx->buf);


    g_free(ctx);

}