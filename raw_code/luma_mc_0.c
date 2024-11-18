static void luma_mc(HEVCContext *s, int16_t *dst, ptrdiff_t dststride,

                    AVFrame *ref, const Mv *mv, int x_off, int y_off,

                    int block_w, int block_h)

{

    HEVCLocalContext *lc = &s->HEVClc;

    uint8_t *src         = ref->data[0];

    ptrdiff_t srcstride  = ref->linesize[0];

    int pic_width        = s->ps.sps->width;

    int pic_height       = s->ps.sps->height;



    int mx         = mv->x & 3;

    int my         = mv->y & 3;

    int extra_left = ff_hevc_qpel_extra_before[mx];

    int extra_top  = ff_hevc_qpel_extra_before[my];



    x_off += mv->x >> 2;

    y_off += mv->y >> 2;

    src   += y_off * srcstride + (x_off << s->ps.sps->pixel_shift);



    if (x_off < extra_left || y_off < extra_top ||

        x_off >= pic_width - block_w - ff_hevc_qpel_extra_after[mx] ||

        y_off >= pic_height - block_h - ff_hevc_qpel_extra_after[my]) {

        const int edge_emu_stride = EDGE_EMU_BUFFER_STRIDE << s->ps.sps->pixel_shift;

        int offset = extra_top * srcstride + (extra_left << s->ps.sps->pixel_shift);

        int buf_offset = extra_top *

                         edge_emu_stride + (extra_left << s->ps.sps->pixel_shift);



        s->vdsp.emulated_edge_mc(lc->edge_emu_buffer, src - offset,

                                 edge_emu_stride, srcstride,

                                 block_w + ff_hevc_qpel_extra[mx],

                                 block_h + ff_hevc_qpel_extra[my],

                                 x_off - extra_left, y_off - extra_top,

                                 pic_width, pic_height);

        src = lc->edge_emu_buffer + buf_offset;

        srcstride = edge_emu_stride;

    }

    s->hevcdsp.put_hevc_qpel[my][mx](dst, dststride, src, srcstride, block_w,

                                     block_h, lc->mc_buffer);

}
