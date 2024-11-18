static int ljpeg_encode_yuv(AVCodecContext *avctx, PutBitContext *pb,

                            const AVFrame *frame)

{

    const int predictor = avctx->prediction_method + 1;

    LJpegEncContext *s  = avctx->priv_data;

    const int mb_width  = (avctx->width  + s->hsample[0] - 1) / s->hsample[0];

    const int mb_height = (avctx->height + s->vsample[0] - 1) / s->vsample[0];

    int mb_x, mb_y;



    for (mb_y = 0; mb_y < mb_height; mb_y++) {

        if (pb->buf_end - pb->buf - (put_bits_count(pb) >> 3) <

            mb_width * 4 * 3 * s->hsample[0] * s->vsample[0]) {

            av_log(avctx, AV_LOG_ERROR, "encoded frame too large\n");

            return -1;

        }



        for (mb_x = 0; mb_x < mb_width; mb_x++)

            ljpeg_encode_yuv_mb(s, pb, frame, predictor, mb_x, mb_y);

    }



    return 0;

}
