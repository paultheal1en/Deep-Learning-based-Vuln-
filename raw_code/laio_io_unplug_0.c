void laio_io_unplug(BlockDriverState *bs, void *aio_ctx, bool unplug)

{

    struct qemu_laio_state *s = aio_ctx;



    assert(s->io_q.plugged > 0 || !unplug);



    if (unplug && --s->io_q.plugged > 0) {

        return;

    }



    if (!s->io_q.blocked && !QSIMPLEQ_EMPTY(&s->io_q.pending)) {

        ioq_submit(s);

    }

}
