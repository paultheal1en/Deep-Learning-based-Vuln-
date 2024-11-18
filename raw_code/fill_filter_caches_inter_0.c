static av_always_inline void fill_filter_caches_inter(const H264Context *h,

                                                      H264SliceContext *sl,

                                                      int mb_type, int top_xy,

                                                      int left_xy[LEFT_MBS],

                                                      int top_type,

                                                      int left_type[LEFT_MBS],

                                                      int mb_xy, int list)

{

    int b_stride = h->b_stride;

    int16_t(*mv_dst)[2] = &sl->mv_cache[list][scan8[0]];

    int8_t *ref_cache   = &sl->ref_cache[list][scan8[0]];

    if (IS_INTER(mb_type) || IS_DIRECT(mb_type)) {

        if (USES_LIST(top_type, list)) {

            const int b_xy  = h->mb2b_xy[top_xy] + 3 * b_stride;

            const int b8_xy = 4 * top_xy + 2;

            int (*ref2frm)[64] = sl->ref2frm[h->slice_table[top_xy] & (MAX_SLICES - 1)][0] + (MB_MBAFF(sl) ? 20 : 2);

            AV_COPY128(mv_dst - 1 * 8, h->cur_pic.motion_val[list][b_xy + 0]);

            ref_cache[0 - 1 * 8] =

            ref_cache[1 - 1 * 8] = ref2frm[list][h->cur_pic.ref_index[list][b8_xy + 0]];

            ref_cache[2 - 1 * 8] =

            ref_cache[3 - 1 * 8] = ref2frm[list][h->cur_pic.ref_index[list][b8_xy + 1]];

        } else {

            AV_ZERO128(mv_dst - 1 * 8);

            AV_WN32A(&ref_cache[0 - 1 * 8], ((LIST_NOT_USED) & 0xFF) * 0x01010101u);

        }



        if (!IS_INTERLACED(mb_type ^ left_type[LTOP])) {

            if (USES_LIST(left_type[LTOP], list)) {

                const int b_xy  = h->mb2b_xy[left_xy[LTOP]] + 3;

                const int b8_xy = 4 * left_xy[LTOP] + 1;

                int (*ref2frm)[64] = sl->ref2frm[h->slice_table[left_xy[LTOP]] & (MAX_SLICES - 1)][0] + (MB_MBAFF(sl) ? 20 : 2);

                AV_COPY32(mv_dst - 1 +  0, h->cur_pic.motion_val[list][b_xy + b_stride * 0]);

                AV_COPY32(mv_dst - 1 +  8, h->cur_pic.motion_val[list][b_xy + b_stride * 1]);

                AV_COPY32(mv_dst - 1 + 16, h->cur_pic.motion_val[list][b_xy + b_stride * 2]);

                AV_COPY32(mv_dst - 1 + 24, h->cur_pic.motion_val[list][b_xy + b_stride * 3]);

                ref_cache[-1 +  0] =

                ref_cache[-1 +  8] = ref2frm[list][h->cur_pic.ref_index[list][b8_xy + 2 * 0]];

                ref_cache[-1 + 16] =

                ref_cache[-1 + 24] = ref2frm[list][h->cur_pic.ref_index[list][b8_xy + 2 * 1]];

            } else {

                AV_ZERO32(mv_dst - 1 +  0);

                AV_ZERO32(mv_dst - 1 +  8);

                AV_ZERO32(mv_dst - 1 + 16);

                AV_ZERO32(mv_dst - 1 + 24);

                ref_cache[-1 +  0] =

                ref_cache[-1 +  8] =

                ref_cache[-1 + 16] =

                ref_cache[-1 + 24] = LIST_NOT_USED;

            }

        }

    }



    if (!USES_LIST(mb_type, list)) {

        fill_rectangle(mv_dst, 4, 4, 8, pack16to32(0, 0), 4);

        AV_WN32A(&ref_cache[0 * 8], ((LIST_NOT_USED) & 0xFF) * 0x01010101u);

        AV_WN32A(&ref_cache[1 * 8], ((LIST_NOT_USED) & 0xFF) * 0x01010101u);

        AV_WN32A(&ref_cache[2 * 8], ((LIST_NOT_USED) & 0xFF) * 0x01010101u);

        AV_WN32A(&ref_cache[3 * 8], ((LIST_NOT_USED) & 0xFF) * 0x01010101u);

        return;

    }



    {

        int8_t *ref = &h->cur_pic.ref_index[list][4 * mb_xy];

        int (*ref2frm)[64] = sl->ref2frm[sl->slice_num & (MAX_SLICES - 1)][0] + (MB_MBAFF(sl) ? 20 : 2);

        uint32_t ref01 = (pack16to32(ref2frm[list][ref[0]], ref2frm[list][ref[1]]) & 0x00FF00FF) * 0x0101;

        uint32_t ref23 = (pack16to32(ref2frm[list][ref[2]], ref2frm[list][ref[3]]) & 0x00FF00FF) * 0x0101;

        AV_WN32A(&ref_cache[0 * 8], ref01);

        AV_WN32A(&ref_cache[1 * 8], ref01);

        AV_WN32A(&ref_cache[2 * 8], ref23);

        AV_WN32A(&ref_cache[3 * 8], ref23);

    }



    {

        int16_t(*mv_src)[2] = &h->cur_pic.motion_val[list][4 * sl->mb_x + 4 * sl->mb_y * b_stride];

        AV_COPY128(mv_dst + 8 * 0, mv_src + 0 * b_stride);

        AV_COPY128(mv_dst + 8 * 1, mv_src + 1 * b_stride);

        AV_COPY128(mv_dst + 8 * 2, mv_src + 2 * b_stride);

        AV_COPY128(mv_dst + 8 * 3, mv_src + 3 * b_stride);

    }

}
