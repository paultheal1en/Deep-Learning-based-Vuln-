static av_always_inline void dnxhd_decode_dct_block(DNXHDContext *ctx,

                                                    DCTELEM *block, int n,

                                                    int qscale,

                                                    int index_bits,

                                                    int level_bias,

                                                    int level_shift)

{

    int i, j, index1, index2, len;

    int level, component, sign;

    const uint8_t *weight_matrix;

    OPEN_READER(bs, &ctx->gb);



    if (n&2) {

        component = 1 + (n&1);

        weight_matrix = ctx->cid_table->chroma_weight;

    } else {

        component = 0;

        weight_matrix = ctx->cid_table->luma_weight;

    }



    UPDATE_CACHE(bs, &ctx->gb);

    GET_VLC(len, bs, &ctx->gb, ctx->dc_vlc.table, DNXHD_DC_VLC_BITS, 1);

    if (len) {

        level = GET_CACHE(bs, &ctx->gb);

        LAST_SKIP_BITS(bs, &ctx->gb, len);

        sign  = ~level >> 31;

        level = (NEG_USR32(sign ^ level, len) ^ sign) - sign;

        ctx->last_dc[component] += level;

    }

    block[0] = ctx->last_dc[component];

    //av_log(ctx->avctx, AV_LOG_DEBUG, "dc %d\n", block[0]);



    for (i = 1; ; i++) {

        UPDATE_CACHE(bs, &ctx->gb);

        GET_VLC(index1, bs, &ctx->gb, ctx->ac_vlc.table,

                DNXHD_VLC_BITS, 2);

        //av_log(ctx->avctx, AV_LOG_DEBUG, "index %d\n", index1);

        level = ctx->cid_table->ac_level[index1];

        if (!level) { /* EOB */

            //av_log(ctx->avctx, AV_LOG_DEBUG, "EOB\n");

            break;

        }



        sign = SHOW_SBITS(bs, &ctx->gb, 1);

        SKIP_BITS(bs, &ctx->gb, 1);



        if (ctx->cid_table->ac_index_flag[index1]) {

            level += SHOW_UBITS(bs, &ctx->gb, index_bits) << 6;

            SKIP_BITS(bs, &ctx->gb, index_bits);

        }



        if (ctx->cid_table->ac_run_flag[index1]) {

            UPDATE_CACHE(bs, &ctx->gb);

            GET_VLC(index2, bs, &ctx->gb, ctx->run_vlc.table,

                    DNXHD_VLC_BITS, 2);

            i += ctx->cid_table->run[index2];

        }



        if (i > 63) {

            av_log(ctx->avctx, AV_LOG_ERROR, "ac tex damaged %d, %d\n", n, i);

            break;

        }



        j = ctx->scantable.permutated[i];

        //av_log(ctx->avctx, AV_LOG_DEBUG, "j %d\n", j);

        //av_log(ctx->avctx, AV_LOG_DEBUG, "level %d, weight %d\n", level, weight_matrix[i]);

        level = (2*level+1) * qscale * weight_matrix[i];

        if (level_bias < 32 || weight_matrix[i] != level_bias)

            level += level_bias;

        level >>= level_shift;



        //av_log(NULL, AV_LOG_DEBUG, "i %d, j %d, end level %d\n", i, j, level);

        block[j] = (level^sign) - sign;

    }



    CLOSE_READER(bs, &ctx->gb);

}
