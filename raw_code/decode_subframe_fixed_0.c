static int decode_subframe_fixed(FLACContext *s, int channel, int pred_order)

{

    int i;

        

    av_log(s->avctx, AV_LOG_DEBUG, "  SUBFRAME FIXED\n");

        

    /* warm up samples */

    av_log(s->avctx, AV_LOG_DEBUG, "   warm up samples: %d\n", pred_order);

        

    for (i = 0; i < pred_order; i++)

    {

        s->decoded[channel][i] = get_sbits(&s->gb, s->curr_bps);

//        av_log(s->avctx, AV_LOG_DEBUG, "    %d: %d\n", i, s->decoded[channel][i]);

    }

    

    if (decode_residuals(s, channel, pred_order) < 0)

        return -1;



    switch(pred_order)

    {

        case 0:

            break;

        case 1:

            for (i = pred_order; i < s->blocksize; i++)

                s->decoded[channel][i] +=   s->decoded[channel][i-1];

            break;

        case 2:

            for (i = pred_order; i < s->blocksize; i++)

                s->decoded[channel][i] += 2*s->decoded[channel][i-1]

                                          - s->decoded[channel][i-2];

            break;

        case 3:

            for (i = pred_order; i < s->blocksize; i++)

                s->decoded[channel][i] += 3*s->decoded[channel][i-1] 

                                        - 3*s->decoded[channel][i-2]

                                        +   s->decoded[channel][i-3];

            break;

        case 4:

            for (i = pred_order; i < s->blocksize; i++)

                s->decoded[channel][i] += 4*s->decoded[channel][i-1] 

                                        - 6*s->decoded[channel][i-2]

                                        + 4*s->decoded[channel][i-3]

                                        -   s->decoded[channel][i-4];

            break;

        default:

            av_log(s->avctx, AV_LOG_ERROR, "illegal pred order %d\n", pred_order);

            return -1;

    }



    return 0;

}
