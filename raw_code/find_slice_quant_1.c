static int find_slice_quant(AVCodecContext *avctx, const AVFrame *pic,

                            int trellis_node, int x, int y, int mbs_per_slice,

                            ProresThreadData *td)

{

    ProresContext *ctx = avctx->priv_data;

    int i, q, pq, xp, yp;

    const uint16_t *src;

    int slice_width_factor = av_log2(mbs_per_slice);

    int num_cblocks[MAX_PLANES], pwidth;

    int plane_factor[MAX_PLANES], is_chroma[MAX_PLANES];

    const int min_quant = ctx->profile_info->min_quant;

    const int max_quant = ctx->profile_info->max_quant;

    int error, bits, bits_limit;

    int mbs, prev, cur, new_score;

    int slice_bits[TRELLIS_WIDTH], slice_score[TRELLIS_WIDTH];

    int overquant;

    uint16_t *qmat;

    int linesize[4], line_add;



    if (ctx->pictures_per_frame == 1)

        line_add = 0;

    else

        line_add = ctx->cur_picture_idx ^ !pic->top_field_first;

    mbs = x + mbs_per_slice;



    for (i = 0; i < ctx->num_planes; i++) {

        is_chroma[i]    = (i == 1 || i == 2);

        plane_factor[i] = slice_width_factor + 2;

        if (is_chroma[i])

            plane_factor[i] += ctx->chroma_factor - 3;

        if (!is_chroma[i] || ctx->chroma_factor == CFACTOR_Y444) {

            xp             = x << 4;

            yp             = y << 4;

            num_cblocks[i] = 4;

            pwidth         = avctx->width;

        } else {

            xp             = x << 3;

            yp             = y << 4;

            num_cblocks[i] = 2;

            pwidth         = avctx->width >> 1;

        }



        linesize[i] = pic->linesize[i] * ctx->pictures_per_frame;

        src = (const uint16_t*)(pic->data[i] + yp * linesize[i] +

                                line_add * pic->linesize[i]) + xp;



        if (i < 3) {

            get_slice_data(ctx, src, linesize[i], xp, yp,

                           pwidth, avctx->height / ctx->pictures_per_frame,

                           td->blocks[i], td->emu_buf,

                           mbs_per_slice, num_cblocks[i], is_chroma[i]);

        } else {

            get_alpha_data(ctx, src, linesize[i], xp, yp,

                           pwidth, avctx->height / ctx->pictures_per_frame,

                           td->blocks[i], mbs_per_slice, ctx->alpha_bits);

        }

    }



    for (q = min_quant; q < max_quant + 2; q++) {

        td->nodes[trellis_node + q].prev_node = -1;

        td->nodes[trellis_node + q].quant     = q;

    }



    // todo: maybe perform coarser quantising to fit into frame size when needed

    for (q = min_quant; q <= max_quant; q++) {

        bits  = 0;

        error = 0;

        for (i = 0; i < ctx->num_planes - !!ctx->alpha_bits; i++) {

            bits += estimate_slice_plane(ctx, &error, i,

                                         src, linesize[i],

                                         mbs_per_slice,

                                         num_cblocks[i], plane_factor[i],

                                         ctx->quants[q], td);

        }

        if (ctx->alpha_bits)

            bits += estimate_alpha_plane(ctx, &error, src, linesize[3],

                                         mbs_per_slice, q, td->blocks[3]);

        if (bits > 65000 * 8) {

            error = SCORE_LIMIT;

            break;

        }

        slice_bits[q]  = bits;

        slice_score[q] = error;

    }

    if (slice_bits[max_quant] <= ctx->bits_per_mb * mbs_per_slice) {

        slice_bits[max_quant + 1]  = slice_bits[max_quant];

        slice_score[max_quant + 1] = slice_score[max_quant] + 1;

        overquant = max_quant;

    } else {

        for (q = max_quant + 1; q < 128; q++) {

            bits  = 0;

            error = 0;

            if (q < MAX_STORED_Q) {

                qmat = ctx->quants[q];

            } else {

                qmat = td->custom_q;

                for (i = 0; i < 64; i++)

                    qmat[i] = ctx->quant_mat[i] * q;

            }

            for (i = 0; i < ctx->num_planes - !!ctx->alpha_bits; i++) {

                bits += estimate_slice_plane(ctx, &error, i,

                                             src, linesize[i],

                                             mbs_per_slice,

                                             num_cblocks[i], plane_factor[i],

                                             qmat, td);

            }

            if (ctx->alpha_bits)

                bits += estimate_alpha_plane(ctx, &error, src, linesize[3],

                                             mbs_per_slice, q, td->blocks[3]);

            if (bits <= ctx->bits_per_mb * mbs_per_slice)

                break;

        }



        slice_bits[max_quant + 1]  = bits;

        slice_score[max_quant + 1] = error;

        overquant = q;

    }

    td->nodes[trellis_node + max_quant + 1].quant = overquant;



    bits_limit = mbs * ctx->bits_per_mb;

    for (pq = min_quant; pq < max_quant + 2; pq++) {

        prev = trellis_node - TRELLIS_WIDTH + pq;



        for (q = min_quant; q < max_quant + 2; q++) {

            cur = trellis_node + q;



            bits  = td->nodes[prev].bits + slice_bits[q];

            error = slice_score[q];

            if (bits > bits_limit)

                error = SCORE_LIMIT;



            if (td->nodes[prev].score < SCORE_LIMIT && error < SCORE_LIMIT)

                new_score = td->nodes[prev].score + error;

            else

                new_score = SCORE_LIMIT;

            if (td->nodes[cur].prev_node == -1 ||

                td->nodes[cur].score >= new_score) {



                td->nodes[cur].bits      = bits;

                td->nodes[cur].score     = new_score;

                td->nodes[cur].prev_node = prev;

            }

        }

    }



    error = td->nodes[trellis_node + min_quant].score;

    pq    = trellis_node + min_quant;

    for (q = min_quant + 1; q < max_quant + 2; q++) {

        if (td->nodes[trellis_node + q].score <= error) {

            error = td->nodes[trellis_node + q].score;

            pq    = trellis_node + q;

        }

    }



    return pq;

}
