static void deblocking_filter_CTB(HEVCContext *s, int x0, int y0)

{

    uint8_t *src;

    int x, y, x_end, y_end, chroma;

    int c_tc[2], beta[2], tc[2];

    uint8_t no_p[2] = { 0 };

    uint8_t no_q[2] = { 0 };



    int log2_ctb_size = s->sps->log2_ctb_size;

    int ctb_size        = 1 << log2_ctb_size;

    int ctb             = (x0 >> log2_ctb_size) +

                          (y0 >> log2_ctb_size) * s->sps->ctb_width;

    int cur_tc_offset   = s->deblock[ctb].tc_offset;

    int cur_beta_offset = s->deblock[ctb].beta_offset;

    int tc_offset, left_tc_offset, beta_offset, left_beta_offset;

    int pcmf = (s->sps->pcm_enabled_flag &&

                s->sps->pcm.loop_filter_disable_flag) ||

               s->pps->transquant_bypass_enable_flag;



    if (x0) {

        left_tc_offset   = s->deblock[ctb - 1].tc_offset;

        left_beta_offset = s->deblock[ctb - 1].beta_offset;

    }



    x_end = x0 + ctb_size;

    if (x_end > s->sps->width)

        x_end = s->sps->width;

    y_end = y0 + ctb_size;

    if (y_end > s->sps->height)

        y_end = s->sps->height;



    tc_offset   = cur_tc_offset;

    beta_offset = cur_beta_offset;



    // vertical filtering luma

    for (y = y0; y < y_end; y += 8) {

        for (x = x0 ? x0 : 8; x < x_end; x += 8) {

            const int bs0 = s->vertical_bs[(x >> 3) + (y       >> 2) * s->bs_width];

            const int bs1 = s->vertical_bs[(x >> 3) + ((y + 4) >> 2) * s->bs_width];

            if (bs0 || bs1) {

                const int qp0 = (get_qPy(s, x - 1, y)     + get_qPy(s, x, y)     + 1) >> 1;

                const int qp1 = (get_qPy(s, x - 1, y + 4) + get_qPy(s, x, y + 4) + 1) >> 1;



                beta[0] = betatable[av_clip(qp0 + beta_offset, 0, MAX_QP)];

                beta[1] = betatable[av_clip(qp1 + beta_offset, 0, MAX_QP)];

                tc[0]   = bs0 ? TC_CALC(qp0, bs0) : 0;

                tc[1]   = bs1 ? TC_CALC(qp1, bs1) : 0;

                src     = &s->frame->data[LUMA][y * s->frame->linesize[LUMA] + (x << s->sps->pixel_shift)];

                if (pcmf) {

                    no_p[0] = get_pcm(s, x - 1, y);

                    no_p[1] = get_pcm(s, x - 1, y + 4);

                    no_q[0] = get_pcm(s, x, y);

                    no_q[1] = get_pcm(s, x, y + 4);

                    s->hevcdsp.hevc_v_loop_filter_luma_c(src,

                                                         s->frame->linesize[LUMA],

                                                         beta, tc, no_p, no_q);

                } else

                    s->hevcdsp.hevc_v_loop_filter_luma(src,

                                                       s->frame->linesize[LUMA],

                                                       beta, tc, no_p, no_q);

            }

        }

    }



    // vertical filtering chroma

    for (chroma = 1; chroma <= 2; chroma++) {

        for (y = y0; y < y_end; y += 16) {

            for (x = x0 ? x0 : 16; x < x_end; x += 16) {

                const int bs0 = s->vertical_bs[(x >> 3) + (y       >> 2) * s->bs_width];

                const int bs1 = s->vertical_bs[(x >> 3) + ((y + 8) >> 2) * s->bs_width];

                if ((bs0 == 2) || (bs1 == 2)) {

                    const int qp0 = (get_qPy(s, x - 1, y)     + get_qPy(s, x, y)     + 1) >> 1;

                    const int qp1 = (get_qPy(s, x - 1, y + 8) + get_qPy(s, x, y + 8) + 1) >> 1;



                    c_tc[0] = (bs0 == 2) ? chroma_tc(s, qp0, chroma, tc_offset) : 0;

                    c_tc[1] = (bs1 == 2) ? chroma_tc(s, qp1, chroma, tc_offset) : 0;

                    src     = &s->frame->data[chroma][y / 2 * s->frame->linesize[chroma] + ((x / 2) << s->sps->pixel_shift)];

                    if (pcmf) {

                        no_p[0] = get_pcm(s, x - 1, y);

                        no_p[1] = get_pcm(s, x - 1, y + 8);

                        no_q[0] = get_pcm(s, x, y);

                        no_q[1] = get_pcm(s, x, y + 8);

                        s->hevcdsp.hevc_v_loop_filter_chroma_c(src,

                                                               s->frame->linesize[chroma],

                                                               c_tc, no_p, no_q);

                    } else

                        s->hevcdsp.hevc_v_loop_filter_chroma(src,

                                                             s->frame->linesize[chroma],

                                                             c_tc, no_p, no_q);

                }

            }

        }

    }



    // horizontal filtering luma

    if (x_end != s->sps->width)

        x_end -= 8;

    for (y = y0 ? y0 : 8; y < y_end; y += 8) {

        for (x = x0 ? x0 - 8 : 0; x < x_end; x += 8) {

            const int bs0 = s->horizontal_bs[(x +     y * s->bs_width) >> 2];

            const int bs1 = s->horizontal_bs[(x + 4 + y * s->bs_width) >> 2];

            if (bs0 || bs1) {

                const int qp0 = (get_qPy(s, x, y - 1)     + get_qPy(s, x, y)     + 1) >> 1;

                const int qp1 = (get_qPy(s, x + 4, y - 1) + get_qPy(s, x + 4, y) + 1) >> 1;



                tc_offset   = x >= x0 ? cur_tc_offset : left_tc_offset;

                beta_offset = x >= x0 ? cur_beta_offset : left_beta_offset;



                beta[0] = betatable[av_clip(qp0 + beta_offset, 0, MAX_QP)];

                beta[1] = betatable[av_clip(qp1 + beta_offset, 0, MAX_QP)];

                tc[0]   = bs0 ? TC_CALC(qp0, bs0) : 0;

                tc[1]   = bs1 ? TC_CALC(qp1, bs1) : 0;

                src     = &s->frame->data[LUMA][y * s->frame->linesize[LUMA] + (x << s->sps->pixel_shift)];

                if (pcmf) {

                    no_p[0] = get_pcm(s, x, y - 1);

                    no_p[1] = get_pcm(s, x + 4, y - 1);

                    no_q[0] = get_pcm(s, x, y);

                    no_q[1] = get_pcm(s, x + 4, y);

                    s->hevcdsp.hevc_h_loop_filter_luma_c(src,

                                                         s->frame->linesize[LUMA],

                                                         beta, tc, no_p, no_q);

                } else

                    s->hevcdsp.hevc_h_loop_filter_luma(src,

                                                       s->frame->linesize[LUMA],

                                                       beta, tc, no_p, no_q);

            }

        }

    }



    // horizontal filtering chroma

    for (chroma = 1; chroma <= 2; chroma++) {

        for (y = y0 ? y0 : 16; y < y_end; y += 16) {

            for (x = x0 - 8; x < x_end; x += 16) {

                int bs0, bs1;

                // to make sure no memory access over boundary when x = -8

                // TODO: simplify with row based deblocking

                if (x < 0) {

                    bs0 = 0;

                    bs1 = s->horizontal_bs[(x + 8 + y * s->bs_width) >> 2];

                } else if (x >= x_end - 8) {

                    bs0 = s->horizontal_bs[(x +     y * s->bs_width) >> 2];

                    bs1 = 0;

                } else {

                    bs0 = s->horizontal_bs[(x + y     * s->bs_width) >> 2];

                    bs1 = s->horizontal_bs[(x + 8 + y * s->bs_width) >> 2];

                }



                if ((bs0 == 2) || (bs1 == 2)) {

                    const int qp0 = bs0 == 2 ? (get_qPy(s, x,     y - 1) + get_qPy(s, x,     y) + 1) >> 1 : 0;

                    const int qp1 = bs1 == 2 ? (get_qPy(s, x + 8, y - 1) + get_qPy(s, x + 8, y) + 1) >> 1 : 0;



                    tc_offset = x >= x0 ? cur_tc_offset : left_tc_offset;

                    c_tc[0]   = bs0 == 2 ? chroma_tc(s, qp0, chroma, tc_offset)     : 0;

                    c_tc[1]   = bs1 == 2 ? chroma_tc(s, qp1, chroma, cur_tc_offset) : 0;

                    src       = &s->frame->data[chroma][y / 2 * s->frame->linesize[chroma] + ((x / 2) << s->sps->pixel_shift)];

                    if (pcmf) {

                        no_p[0] = get_pcm(s, x, y - 1);

                        no_p[1] = get_pcm(s, x + 8, y - 1);

                        no_q[0] = get_pcm(s, x, y);

                        no_q[1] = get_pcm(s, x + 8, y);

                        s->hevcdsp.hevc_h_loop_filter_chroma_c(src,

                                                               s->frame->linesize[chroma],

                                                               c_tc, no_p, no_q);

                    } else

                        s->hevcdsp.hevc_h_loop_filter_chroma(src,

                                                             s->frame->linesize[chroma],

                                                             c_tc, no_p, no_q);

                }

            }

        }

    }

}
