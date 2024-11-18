static int atrac1_decode_frame(AVCodecContext *avctx, void *data,

                               int *data_size, AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size       = avpkt->size;

    AT1Ctx *q          = avctx->priv_data;

    int ch, ret, i;

    GetBitContext gb;

    float* samples = data;





    if (buf_size < 212 * q->channels) {

        av_log(q,AV_LOG_ERROR,"Not enought data to decode!\n");

        return -1;

    }



    for (ch = 0; ch < q->channels; ch++) {

        AT1SUCtx* su = &q->SUs[ch];



        init_get_bits(&gb, &buf[212 * ch], 212 * 8);



        /* parse block_size_mode, 1st byte */

        ret = at1_parse_bsm(&gb, su->log2_block_count);

        if (ret < 0)

            return ret;



        ret = at1_unpack_dequant(&gb, su, q->spec);

        if (ret < 0)

            return ret;



        ret = at1_imdct_block(su, q);

        if (ret < 0)

            return ret;

        at1_subband_synthesis(q, su, q->out_samples[ch]);

    }



    /* interleave; FIXME, should create/use a DSP function */

    if (q->channels == 1) {

        /* mono */

        memcpy(samples, q->out_samples[0], AT1_SU_SAMPLES * 4);

    } else {

        /* stereo */

        for (i = 0; i < AT1_SU_SAMPLES; i++) {

            samples[i * 2]     = q->out_samples[0][i];

            samples[i * 2 + 1] = q->out_samples[1][i];

        }

    }



    *data_size = q->channels * AT1_SU_SAMPLES * sizeof(*samples);

    return avctx->block_align;

}
