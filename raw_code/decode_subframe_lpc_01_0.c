static int decode_subframe_lpc(FLACContext *s, int channel, int pred_order)

{

    int sum, i, j;

    int coeff_prec, qlevel;

    int coeffs[pred_order];

        

//    av_log(s->avctx, AV_LOG_DEBUG, "  SUBFRAME LPC\n");

        

    /* warm up samples */

//    av_log(s->avctx, AV_LOG_DEBUG, "   warm up samples: %d\n", pred_order);

        

    for (i = 0; i < pred_order; i++)

    {

        s->decoded[channel][i] = get_sbits(&s->gb, s->curr_bps);

//        av_log(s->avctx, AV_LOG_DEBUG, "    %d: %d\n", i, s->decoded[channel][i]);

    }

    

    coeff_prec = get_bits(&s->gb, 4) + 1;

    if (coeff_prec == 16)

    {

        av_log(s->avctx, AV_LOG_DEBUG, "invalid coeff precision\n");

        return -1;

    }

    av_log(s->avctx, AV_LOG_DEBUG, "   qlp coeff prec: %d\n", coeff_prec);

    qlevel = get_sbits(&s->gb, 5);

    av_log(s->avctx, AV_LOG_DEBUG, "   quant level: %d\n", qlevel);

assert(qlevel >= 0); //FIXME

    for (i = 0; i < pred_order; i++)

    {

        coeffs[i] = get_sbits(&s->gb, coeff_prec);

//        av_log(s->avctx, AV_LOG_DEBUG, "    %d: %d\n", i, coeffs[i]);

    }

    

    if (decode_residuals(s, channel, pred_order) < 0)

        return -1;



    for (i = pred_order; i < s->blocksize; i++)

    {

        sum = 0;

        for (j = 0; j < pred_order; j++)

            sum += coeffs[j] * s->decoded[channel][i-j-1];

        s->decoded[channel][i] += sum >> qlevel;

    }

    

    return 0;

}
