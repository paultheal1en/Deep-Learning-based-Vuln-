static void vc1_put_signed_blocks_clamped(VC1Context *v)

{

    MpegEncContext *s = &v->s;

    int topleft_mb_pos, top_mb_pos;

    int stride_y, fieldtx;

    int v_dist;



    /* The put pixels loop is always one MB row behind the decoding loop,

     * because we can only put pixels when overlap filtering is done, and

     * for filtering of the bottom edge of a MB, we need the next MB row

     * present as well.

     * Within the row, the put pixels loop is also one MB col behind the

     * decoding loop. The reason for this is again, because for filtering

     * of the right MB edge, we need the next MB present. */

    if (!s->first_slice_line) {

        if (s->mb_x) {

            topleft_mb_pos = (s->mb_y - 1) * s->mb_stride + s->mb_x - 1;

            fieldtx        = v->fieldtx_plane[topleft_mb_pos];

            stride_y       = s->linesize << fieldtx;

            v_dist         = (16 - fieldtx) >> (fieldtx == 0);

            s->dsp.put_signed_pixels_clamped(v->block[v->topleft_blk_idx][0],

                                             s->dest[0] - 16 * s->linesize - 16,

                                             stride_y);

            s->dsp.put_signed_pixels_clamped(v->block[v->topleft_blk_idx][1],

                                             s->dest[0] - 16 * s->linesize - 8,

                                             stride_y);

            s->dsp.put_signed_pixels_clamped(v->block[v->topleft_blk_idx][2],

                                             s->dest[0] - v_dist * s->linesize - 16,

                                             stride_y);

            s->dsp.put_signed_pixels_clamped(v->block[v->topleft_blk_idx][3],

                                             s->dest[0] - v_dist * s->linesize - 8,

                                             stride_y);

            s->dsp.put_signed_pixels_clamped(v->block[v->topleft_blk_idx][4],

                                             s->dest[1] - 8 * s->uvlinesize - 8,

                                             s->uvlinesize);

            s->dsp.put_signed_pixels_clamped(v->block[v->topleft_blk_idx][5],

                                             s->dest[2] - 8 * s->uvlinesize - 8,

                                             s->uvlinesize);

        }

        if (s->mb_x == s->mb_width - 1) {

            top_mb_pos = (s->mb_y - 1) * s->mb_stride + s->mb_x;

            fieldtx    = v->fieldtx_plane[top_mb_pos];

            stride_y   = s->linesize << fieldtx;

            v_dist     = fieldtx ? 15 : 8;

            s->dsp.put_signed_pixels_clamped(v->block[v->top_blk_idx][0],

                                             s->dest[0] - 16 * s->linesize,

                                             stride_y);

            s->dsp.put_signed_pixels_clamped(v->block[v->top_blk_idx][1],

                                             s->dest[0] - 16 * s->linesize + 8,

                                             stride_y);

            s->dsp.put_signed_pixels_clamped(v->block[v->top_blk_idx][2],

                                             s->dest[0] - v_dist * s->linesize,

                                             stride_y);

            s->dsp.put_signed_pixels_clamped(v->block[v->top_blk_idx][3],

                                             s->dest[0] - v_dist * s->linesize + 8,

                                             stride_y);

            s->dsp.put_signed_pixels_clamped(v->block[v->top_blk_idx][4],

                                             s->dest[1] - 8 * s->uvlinesize,

                                             s->uvlinesize);

            s->dsp.put_signed_pixels_clamped(v->block[v->top_blk_idx][5],

                                             s->dest[2] - 8 * s->uvlinesize,

                                             s->uvlinesize);

        }

    }



#define inc_blk_idx(idx) do { \

        idx++; \

        if (idx >= v->n_allocated_blks) \

            idx = 0; \

    } while (0)



    inc_blk_idx(v->topleft_blk_idx);

    inc_blk_idx(v->top_blk_idx);

    inc_blk_idx(v->left_blk_idx);

    inc_blk_idx(v->cur_blk_idx);

}
