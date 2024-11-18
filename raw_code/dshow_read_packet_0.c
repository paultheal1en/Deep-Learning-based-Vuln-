static int dshow_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    struct dshow_ctx *ctx = s->priv_data;

    AVPacketList *pktl = NULL;



    while (!pktl) {

        WaitForSingleObject(ctx->mutex, INFINITE);

        pktl = ctx->pktl;

        if (pktl) {

            *pkt = pktl->pkt;

            ctx->pktl = ctx->pktl->next;

            av_free(pktl);

            ctx->curbufsize -= pkt->size;

        }

        ResetEvent(ctx->event);

        ReleaseMutex(ctx->mutex);

        if (!pktl) {

            if (s->flags & AVFMT_FLAG_NONBLOCK) {

                return AVERROR(EAGAIN);

            } else {

                WaitForSingleObject(ctx->event, INFINITE);

            }

        }

    }



    return pkt->size;

}
