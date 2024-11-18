void aio_bh_update_timeout(AioContext *ctx, uint32_t *timeout)

{

    QEMUBH *bh;



    for (bh = ctx->first_bh; bh; bh = bh->next) {

        if (!bh->deleted && bh->scheduled) {

            if (bh->idle) {

                /* idle bottom halves will be polled at least

                 * every 10ms */

                *timeout = MIN(10, *timeout);

            } else {

                /* non-idle bottom halves will be executed

                 * immediately */

                *timeout = 0;

                break;

            }

        }

    }

}
