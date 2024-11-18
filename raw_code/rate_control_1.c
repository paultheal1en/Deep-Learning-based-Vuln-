static int rate_control(AVCodecContext *avctx, void *arg)

{

    SliceArgs *slice_dat = arg;

    VC2EncContext *s = slice_dat->ctx;

    const int sx = slice_dat->x;

    const int sy = slice_dat->y;

    int quant_buf[2], bits_buf[2], quant = s->q_start, range = s->q_start/3;

    const int64_t top = slice_dat->bits_ceil;

    const double percent = s->tolerance;

    const double bottom = top - top*(percent/100.0f);

    int bits = count_hq_slice(s, sx, sy, quant);

    range -= range & 1; /* Make it an even number */

    while ((bits > top) || (bits < bottom)) {

        range *= bits > top ? +1 : -1;

        quant = av_clip(quant + range, 0, s->q_ceil);

        bits = count_hq_slice(s, sx, sy, quant);

        range = av_clip(range/2, 1, s->q_ceil);

        if (quant_buf[1] == quant) {

            quant = bits_buf[0] < bits ? quant_buf[0] : quant;

            bits = bits_buf[0] < bits ? bits_buf[0] : bits;

            break;

        }

        quant_buf[1] = quant_buf[0];

        quant_buf[0] = quant;

        bits_buf[1] = bits_buf[0];

        bits_buf[0] = bits;

    }

    slice_dat->quant_idx = av_clip(quant, 0, s->q_ceil);

    slice_dat->bytes = FFALIGN((bits >> 3), s->size_scaler) + 4 + s->prefix_bytes;



    return 0;

}
