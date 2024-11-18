static int ff_vp56_decode_mbs(AVCodecContext *avctx, void *data,

                              int jobnr, int threadnr)

{

    VP56Context *s0 = avctx->priv_data;

    int is_alpha = (jobnr == 1);

    VP56Context *s = is_alpha ? s0->alpha_context : s0;

    AVFrame *const p = s->frames[VP56_FRAME_CURRENT];

    int mb_row, mb_col, mb_row_flip, mb_offset = 0;

    int block, y, uv;

    ptrdiff_t stride_y, stride_uv;

    int res;

    int damaged = 0;



    if (p->key_frame) {

        p->pict_type = AV_PICTURE_TYPE_I;

        s->default_models_init(s);

        for (block=0; block<s->mb_height*s->mb_width; block++)

            s->macroblocks[block].type = VP56_MB_INTRA;

    } else {

        p->pict_type = AV_PICTURE_TYPE_P;

        vp56_parse_mb_type_models(s);

        s->parse_vector_models(s);

        s->mb_type = VP56_MB_INTER_NOVEC_PF;

    }



    if (s->parse_coeff_models(s))

        goto next;



    memset(s->prev_dc, 0, sizeof(s->prev_dc));

    s->prev_dc[1][VP56_FRAME_CURRENT] = 128;

    s->prev_dc[2][VP56_FRAME_CURRENT] = 128;



    for (block=0; block < 4*s->mb_width+6; block++) {

        s->above_blocks[block].ref_frame = VP56_FRAME_NONE;

        s->above_blocks[block].dc_coeff = 0;

        s->above_blocks[block].not_null_dc = 0;

    }

    s->above_blocks[2*s->mb_width + 2].ref_frame = VP56_FRAME_CURRENT;

    s->above_blocks[3*s->mb_width + 4].ref_frame = VP56_FRAME_CURRENT;



    stride_y  = p->linesize[0];

    stride_uv = p->linesize[1];



    if (s->flip < 0)

        mb_offset = 7;



    /* main macroblocks loop */

    for (mb_row=0; mb_row<s->mb_height; mb_row++) {

        if (s->flip < 0)

            mb_row_flip = s->mb_height - mb_row - 1;

        else

            mb_row_flip = mb_row;



        for (block=0; block<4; block++) {

            s->left_block[block].ref_frame = VP56_FRAME_NONE;

            s->left_block[block].dc_coeff = 0;

            s->left_block[block].not_null_dc = 0;

        }

        memset(s->coeff_ctx, 0, sizeof(s->coeff_ctx));

        memset(s->coeff_ctx_last, 24, sizeof(s->coeff_ctx_last));



        s->above_block_idx[0] = 1;

        s->above_block_idx[1] = 2;

        s->above_block_idx[2] = 1;

        s->above_block_idx[3] = 2;

        s->above_block_idx[4] = 2*s->mb_width + 2 + 1;

        s->above_block_idx[5] = 3*s->mb_width + 4 + 1;



        s->block_offset[s->frbi] = (mb_row_flip*16 + mb_offset) * stride_y;

        s->block_offset[s->srbi] = s->block_offset[s->frbi] + 8*stride_y;

        s->block_offset[1] = s->block_offset[0] + 8;

        s->block_offset[3] = s->block_offset[2] + 8;

        s->block_offset[4] = (mb_row_flip*8 + mb_offset) * stride_uv;

        s->block_offset[5] = s->block_offset[4];



        for (mb_col=0; mb_col<s->mb_width; mb_col++) {

            if (!damaged) {

                int ret = vp56_decode_mb(s, mb_row, mb_col, is_alpha);

                if (ret < 0)

                    damaged = 1;

            }

            if (damaged)

                vp56_conceal_mb(s, mb_row, mb_col, is_alpha);



            for (y=0; y<4; y++) {

                s->above_block_idx[y] += 2;

                s->block_offset[y] += 16;

            }



            for (uv=4; uv<6; uv++) {

                s->above_block_idx[uv] += 1;

                s->block_offset[uv] += 8;

            }

        }

    }



next:

    if (p->key_frame || s->golden_frame) {

        av_frame_unref(s->frames[VP56_FRAME_GOLDEN]);

        if ((res = av_frame_ref(s->frames[VP56_FRAME_GOLDEN], p)) < 0)

            return res;

    }



    av_frame_unref(s->frames[VP56_FRAME_PREVIOUS]);

    FFSWAP(AVFrame *, s->frames[VP56_FRAME_CURRENT],

                      s->frames[VP56_FRAME_PREVIOUS]);

    return 0;

}
