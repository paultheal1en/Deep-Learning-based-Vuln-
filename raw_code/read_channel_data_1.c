static int read_channel_data(ALSDecContext *ctx, ALSChannelData *cd, int c)

{

    GetBitContext *gb       = &ctx->gb;

    ALSChannelData *current = cd;

    unsigned int channels   = ctx->avctx->channels;

    int entries             = 0;



    while (entries < channels && !(current->stop_flag = get_bits1(gb))) {

        current->master_channel = get_bits_long(gb, av_ceil_log2(channels));



        if (current->master_channel >= channels) {

            av_log(ctx->avctx, AV_LOG_ERROR, "Invalid master channel!\n");

            return -1;

        }



        if (current->master_channel != c) {

            current->time_diff_flag = get_bits1(gb);

            current->weighting[0]   = als_weighting(gb, 1, 16);

            current->weighting[1]   = als_weighting(gb, 2, 14);

            current->weighting[2]   = als_weighting(gb, 1, 16);



            if (current->time_diff_flag) {

                current->weighting[3] = als_weighting(gb, 1, 16);

                current->weighting[4] = als_weighting(gb, 1, 16);

                current->weighting[5] = als_weighting(gb, 1, 16);



                current->time_diff_sign  = get_bits1(gb);

                current->time_diff_index = get_bits(gb, ctx->ltp_lag_length - 3) + 3;

            }

        }



        current++;

        entries++;

    }



    if (entries == channels) {

        av_log(ctx->avctx, AV_LOG_ERROR, "Damaged channel data!\n");

        return -1;

    }



    align_get_bits(gb);

    return 0;

}
