static int cook_decode_frame(AVCodecContext *avctx,

            void *data, int *data_size,

            AVPacket *avpkt) {

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    COOKContext *q = avctx->priv_data;

    int i;

    int offset = 0;

    int chidx = 0;



    if (buf_size < avctx->block_align)

        return buf_size;



    /* estimate subpacket sizes */

    q->subpacket[0].size = avctx->block_align;



    for(i=1;i<q->num_subpackets;i++){

        q->subpacket[i].size = 2 * buf[avctx->block_align - q->num_subpackets + i];

        q->subpacket[0].size -= q->subpacket[i].size + 1;

        if (q->subpacket[0].size < 0) {

            av_log(avctx,AV_LOG_DEBUG,"frame subpacket size total > avctx->block_align!\n");

            return AVERROR_INVALIDDATA;

        }

    }



    /* decode supbackets */

    for(i=0;i<q->num_subpackets;i++){

        q->subpacket[i].bits_per_subpacket = (q->subpacket[i].size*8)>>q->subpacket[i].bits_per_subpdiv;

        q->subpacket[i].ch_idx = chidx;

        av_log(avctx,AV_LOG_DEBUG,"subpacket[%i] size %i js %i %i block_align %i\n",i,q->subpacket[i].size,q->subpacket[i].joint_stereo,offset,avctx->block_align);

        decode_subpacket(q, &q->subpacket[i], buf + offset, data);

        offset += q->subpacket[i].size;

        chidx += q->subpacket[i].num_channels;

        av_log(avctx,AV_LOG_DEBUG,"subpacket[%i] %i %i\n",i,q->subpacket[i].size * 8,get_bits_count(&q->gb));

    }

    *data_size = q->nb_channels * q->samples_per_channel *

                 av_get_bytes_per_sample(avctx->sample_fmt);



    /* Discard the first two frames: no valid audio. */

    if (avctx->frame_number < 2) *data_size = 0;



    return avctx->block_align;

}
