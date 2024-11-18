static int init_context_frame(MpegEncContext *s)

{

    int y_size, c_size, yc_size, i, mb_array_size, mv_table_size, x, y;



    s->mb_width   = (s->width + 15) / 16;

    s->mb_stride  = s->mb_width + 1;

    s->b8_stride  = s->mb_width * 2 + 1;

    s->b4_stride  = s->mb_width * 4 + 1;

    mb_array_size = s->mb_height * s->mb_stride;

    mv_table_size = (s->mb_height + 2) * s->mb_stride + 1;



    /* set default edge pos, will be overriden

     * in decode_header if needed */

    s->h_edge_pos = s->mb_width * 16;

    s->v_edge_pos = s->mb_height * 16;



    s->mb_num     = s->mb_width * s->mb_height;



    s->block_wrap[0] =

    s->block_wrap[1] =

    s->block_wrap[2] =

    s->block_wrap[3] = s->b8_stride;

    s->block_wrap[4] =

    s->block_wrap[5] = s->mb_stride;



    y_size  = s->b8_stride * (2 * s->mb_height + 1);

    c_size  = s->mb_stride * (s->mb_height + 1);

    yc_size = y_size + 2   * c_size;



    if (s->mb_height & 1)

        yc_size += 2*s->b8_stride + 2*s->mb_stride;



    FF_ALLOCZ_OR_GOTO(s->avctx, s->mb_index2xy, (s->mb_num + 1) * sizeof(int), fail); // error ressilience code looks cleaner with this

    for (y = 0; y < s->mb_height; y++)

        for (x = 0; x < s->mb_width; x++)

            s->mb_index2xy[x + y * s->mb_width] = x + y * s->mb_stride;



    s->mb_index2xy[s->mb_height * s->mb_width] = (s->mb_height - 1) * s->mb_stride + s->mb_width; // FIXME really needed?



    if (s->encoding) {

        /* Allocate MV tables */

        FF_ALLOCZ_OR_GOTO(s->avctx, s->p_mv_table_base,                 mv_table_size * 2 * sizeof(int16_t), fail)

        FF_ALLOCZ_OR_GOTO(s->avctx, s->b_forw_mv_table_base,            mv_table_size * 2 * sizeof(int16_t), fail)

        FF_ALLOCZ_OR_GOTO(s->avctx, s->b_back_mv_table_base,            mv_table_size * 2 * sizeof(int16_t), fail)

        FF_ALLOCZ_OR_GOTO(s->avctx, s->b_bidir_forw_mv_table_base,      mv_table_size * 2 * sizeof(int16_t), fail)

        FF_ALLOCZ_OR_GOTO(s->avctx, s->b_bidir_back_mv_table_base,      mv_table_size * 2 * sizeof(int16_t), fail)

        FF_ALLOCZ_OR_GOTO(s->avctx, s->b_direct_mv_table_base,          mv_table_size * 2 * sizeof(int16_t), fail)

        s->p_mv_table            = s->p_mv_table_base + s->mb_stride + 1;

        s->b_forw_mv_table       = s->b_forw_mv_table_base + s->mb_stride + 1;

        s->b_back_mv_table       = s->b_back_mv_table_base + s->mb_stride + 1;

        s->b_bidir_forw_mv_table = s->b_bidir_forw_mv_table_base + s->mb_stride + 1;

        s->b_bidir_back_mv_table = s->b_bidir_back_mv_table_base + s->mb_stride + 1;

        s->b_direct_mv_table     = s->b_direct_mv_table_base + s->mb_stride + 1;



        /* Allocate MB type table */

        FF_ALLOCZ_OR_GOTO(s->avctx, s->mb_type, mb_array_size * sizeof(uint16_t), fail) // needed for encoding



        FF_ALLOCZ_OR_GOTO(s->avctx, s->lambda_table, mb_array_size * sizeof(int), fail)



        FF_ALLOC_OR_GOTO(s->avctx, s->cplx_tab,

                         mb_array_size * sizeof(float), fail);

        FF_ALLOC_OR_GOTO(s->avctx, s->bits_tab,

                         mb_array_size * sizeof(float), fail);



    }



    if (s->codec_id == AV_CODEC_ID_MPEG4 ||

        (s->flags & CODEC_FLAG_INTERLACED_ME)) {

        /* interlaced direct mode decoding tables */

        for (i = 0; i < 2; i++) {

            int j, k;

            for (j = 0; j < 2; j++) {

                for (k = 0; k < 2; k++) {

                    FF_ALLOCZ_OR_GOTO(s->avctx,

                                      s->b_field_mv_table_base[i][j][k],

                                      mv_table_size * 2 * sizeof(int16_t),

                                      fail);

                    s->b_field_mv_table[i][j][k] = s->b_field_mv_table_base[i][j][k] +

                                                   s->mb_stride + 1;

                }

                FF_ALLOCZ_OR_GOTO(s->avctx, s->b_field_select_table [i][j], mb_array_size * 2 * sizeof(uint8_t), fail)

                FF_ALLOCZ_OR_GOTO(s->avctx, s->p_field_mv_table_base[i][j], mv_table_size * 2 * sizeof(int16_t), fail)

                s->p_field_mv_table[i][j] = s->p_field_mv_table_base[i][j] + s->mb_stride + 1;

            }

            FF_ALLOCZ_OR_GOTO(s->avctx, s->p_field_select_table[i], mb_array_size * 2 * sizeof(uint8_t), fail)

        }

    }

    if (s->out_format == FMT_H263) {

        /* cbp values */

        FF_ALLOCZ_OR_GOTO(s->avctx, s->coded_block_base, y_size + (s->mb_height&1)*2*s->b8_stride, fail);

        s->coded_block = s->coded_block_base + s->b8_stride + 1;



        /* cbp, ac_pred, pred_dir */

        FF_ALLOCZ_OR_GOTO(s->avctx, s->cbp_table     , mb_array_size * sizeof(uint8_t), fail);

        FF_ALLOCZ_OR_GOTO(s->avctx, s->pred_dir_table, mb_array_size * sizeof(uint8_t), fail);

    }



    if (s->h263_pred || s->h263_plus || !s->encoding) {

        /* dc values */

        // MN: we need these for  error resilience of intra-frames

        FF_ALLOCZ_OR_GOTO(s->avctx, s->dc_val_base, yc_size * sizeof(int16_t), fail);

        s->dc_val[0] = s->dc_val_base + s->b8_stride + 1;

        s->dc_val[1] = s->dc_val_base + y_size + s->mb_stride + 1;

        s->dc_val[2] = s->dc_val[1] + c_size;

        for (i = 0; i < yc_size; i++)

            s->dc_val_base[i] = 1024;

    }



    /* which mb is a intra block */

    FF_ALLOCZ_OR_GOTO(s->avctx, s->mbintra_table, mb_array_size, fail);

    memset(s->mbintra_table, 1, mb_array_size);



    /* init macroblock skip table */

    FF_ALLOCZ_OR_GOTO(s->avctx, s->mbskip_table, mb_array_size + 2, fail);

    // Note the + 1 is for  a quicker mpeg4 slice_end detection



    return init_er(s);

fail:

    return AVERROR(ENOMEM);

}
