static int roq_encode_frame(AVCodecContext *avctx, AVPacket *pkt,

                            const AVFrame *frame, int *got_packet)

{

    RoqContext *enc = avctx->priv_data;

    int size, ret;



    enc->avctx = avctx;



    enc->frame_to_enc = frame;



    if (frame->quality)

        enc->lambda = frame->quality - 1;

    else

        enc->lambda = 2*ROQ_LAMBDA_SCALE;



    /* 138 bits max per 8x8 block +

     *     256 codebooks*(6 bytes 2x2 + 4 bytes 4x4) + 8 bytes frame header */

    size = ((enc->width * enc->height / 64) * 138 + 7) / 8 + 256 * (6 + 4) + 8;

    if ((ret = ff_alloc_packet(pkt, size)) < 0) {

        av_log(avctx, AV_LOG_ERROR, "Error getting output packet with size %d.\n", size);

        return ret;

    }

    enc->out_buf = pkt->data;



    /* Check for I frame */

    if (enc->framesSinceKeyframe == avctx->gop_size)

        enc->framesSinceKeyframe = 0;



    if (enc->first_frame) {

        /* Alloc memory for the reconstruction data (we must know the stride

         for that) */

        if (ff_get_buffer(avctx, enc->current_frame, 0) ||

            ff_get_buffer(avctx, enc->last_frame, 0)) {

            av_log(avctx, AV_LOG_ERROR, "  RoQ: get_buffer() failed\n");

            return -1;

        }



        /* Before the first video frame, write a "video info" chunk */

        roq_write_video_info_chunk(enc);



        enc->first_frame = 0;

    }



    /* Encode the actual frame */

    roq_encode_video(enc);



    pkt->size   = enc->out_buf - pkt->data;

    if (enc->framesSinceKeyframe == 1)

        pkt->flags |= AV_PKT_FLAG_KEY;

    *got_packet = 1;



    return 0;

}
