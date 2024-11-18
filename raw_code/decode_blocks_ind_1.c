static int decode_blocks_ind(ALSDecContext *ctx, unsigned int ra_frame,

                             unsigned int c, const unsigned int *div_blocks,

                             unsigned int *js_blocks)

{

    unsigned int b;

    ALSBlockData bd = { 0 };



    bd.ra_block         = ra_frame;

    bd.const_block      = ctx->const_block;

    bd.shift_lsbs       = ctx->shift_lsbs;

    bd.opt_order        = ctx->opt_order;

    bd.store_prev_samples = ctx->store_prev_samples;

    bd.use_ltp          = ctx->use_ltp;

    bd.ltp_lag          = ctx->ltp_lag;

    bd.ltp_gain         = ctx->ltp_gain[0];

    bd.quant_cof        = ctx->quant_cof[0];

    bd.lpc_cof          = ctx->lpc_cof[0];

    bd.prev_raw_samples = ctx->prev_raw_samples;

    bd.raw_samples      = ctx->raw_samples[c];





    for (b = 0; b < ctx->num_blocks; b++) {

        bd.block_length     = div_blocks[b];



        if (read_decode_block(ctx, &bd)) {

            // damaged block, write zero for the rest of the frame

            zero_remaining(b, ctx->num_blocks, div_blocks, bd.raw_samples);

            return -1;

        }

        bd.raw_samples += div_blocks[b];

        bd.ra_block     = 0;

    }



    return 0;

}
