static bool run_poll_handlers(AioContext *ctx, int64_t max_ns)

{

    bool progress = false;

    int64_t end_time;



    assert(ctx->notify_me);

    assert(ctx->walking_handlers > 0);

    assert(ctx->poll_disable_cnt == 0);



    trace_run_poll_handlers_begin(ctx, max_ns);



    end_time = qemu_clock_get_ns(QEMU_CLOCK_REALTIME) + max_ns;



    do {

        AioHandler *node;



        QLIST_FOREACH(node, &ctx->aio_handlers, node) {

            if (!node->deleted && node->io_poll &&

                node->io_poll(node->opaque)) {

                progress = true;

            }



            /* Caller handles freeing deleted nodes.  Don't do it here. */

        }

    } while (!progress && qemu_clock_get_ns(QEMU_CLOCK_REALTIME) < end_time);



    trace_run_poll_handlers_end(ctx, progress);



    return progress;

}
