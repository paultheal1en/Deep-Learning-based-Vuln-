static void bench_cb(void *opaque, int ret)

{

    BenchData *b = opaque;

    BlockAIOCB *acb;



    if (ret < 0) {

        error_report("Failed request: %s\n", strerror(-ret));

        exit(EXIT_FAILURE);

    }



    if (b->in_flush) {

        /* Just finished a flush with drained queue: Start next requests */

        assert(b->in_flight == 0);

        b->in_flush = false;

    } else if (b->in_flight > 0) {

        int remaining = b->n - b->in_flight;



        b->n--;

        b->in_flight--;



        /* Time for flush? Drain queue if requested, then flush */

        if (b->flush_interval && remaining % b->flush_interval == 0) {

            if (!b->in_flight || !b->drain_on_flush) {

                BlockCompletionFunc *cb;



                if (b->drain_on_flush) {

                    b->in_flush = true;

                    cb = bench_cb;

                } else {

                    cb = bench_undrained_flush_cb;

                }



                acb = blk_aio_flush(b->blk, cb, b);

                if (!acb) {

                    error_report("Failed to issue flush request");

                    exit(EXIT_FAILURE);

                }

            }

            if (b->drain_on_flush) {

                return;

            }

        }

    }



    while (b->n > b->in_flight && b->in_flight < b->nrreq) {

        if (b->write) {

            acb = blk_aio_pwritev(b->blk, b->offset, b->qiov, 0,

                                  bench_cb, b);

        } else {

            acb = blk_aio_preadv(b->blk, b->offset, b->qiov, 0,

                                 bench_cb, b);

        }

        if (!acb) {

            error_report("Failed to issue request");

            exit(EXIT_FAILURE);

        }

        b->in_flight++;

        b->offset += b->step;

        b->offset %= b->image_size;

    }

}
