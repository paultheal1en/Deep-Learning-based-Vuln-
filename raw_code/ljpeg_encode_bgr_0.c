static int ljpeg_encode_bgr(AVCodecContext *avctx, PutBitContext *pb,

                            const AVFrame *frame)

{

    LJpegEncContext *s    = avctx->priv_data;

    const int width       = frame->width;

    const int height      = frame->height;

    const int linesize    = frame->linesize[0];

    uint16_t (*buffer)[4] = s->scratch;

    const int predictor   = avctx->prediction_method+1;

    int left[3], top[3], topleft[3];

    int x, y, i;



    for (i = 0; i < 3; i++)

        buffer[0][i] = 1 << (9 - 1);



    for (y = 0; y < height; y++) {

        const int modified_predictor = y ? predictor : 1;

        uint8_t *ptr = frame->data[0] + (linesize * y);



        if (pb->buf_end - pb->buf - (put_bits_count(pb) >> 3) < width * 3 * 3) {

            av_log(avctx, AV_LOG_ERROR, "encoded frame too large\n");

            return -1;

        }



        for (i = 0; i < 3; i++)

            top[i]= left[i]= topleft[i]= buffer[0][i];



        for (x = 0; x < width; x++) {

            buffer[x][1] =  ptr[3 * x + 0] -     ptr[3 * x + 1] + 0x100;

            buffer[x][2] =  ptr[3 * x + 2] -     ptr[3 * x + 1] + 0x100;

            buffer[x][0] = (ptr[3 * x + 0] + 2 * ptr[3 * x + 1] + ptr[3 * x + 2]) >> 2;



            for (i = 0; i < 3; i++) {

                int pred, diff;



                PREDICT(pred, topleft[i], top[i], left[i], modified_predictor);



                topleft[i] = top[i];

                top[i]     = buffer[x+1][i];



                left[i]    = buffer[x][i];



                diff       = ((left[i] - pred + 0x100) & 0x1FF) - 0x100;



                if (i == 0)

                    ff_mjpeg_encode_dc(pb, diff, s->huff_size_dc_luminance, s->huff_code_dc_luminance); //FIXME ugly

                else

                    ff_mjpeg_encode_dc(pb, diff, s->huff_size_dc_chrominance, s->huff_code_dc_chrominance);

            }

        }

    }



    return 0;

}
