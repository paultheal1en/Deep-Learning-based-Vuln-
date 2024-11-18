static inline void apply_8x8(MpegEncContext *s,

                             uint8_t *dest_y,

                             uint8_t *dest_cb,

                             uint8_t *dest_cr,

                             int dir,

                             uint8_t **ref_picture,

                             qpel_mc_func (*qpix_op)[16],

                             op_pixels_func (*pix_op)[4])

{

    int dxy, mx, my, src_x, src_y;

    int i;

    int mb_x = s->mb_x;

    int mb_y = s->mb_y;

    uint8_t *ptr, *dest;



    mx = 0;

    my = 0;

    if (s->quarter_sample) {

        for (i = 0; i < 4; i++) {

            int motion_x = s->mv[dir][i][0];

            int motion_y = s->mv[dir][i][1];



            dxy   = ((motion_y & 3) << 2) | (motion_x & 3);

            src_x = mb_x * 16 + (motion_x >> 2) + (i & 1) * 8;

            src_y = mb_y * 16 + (motion_y >> 2) + (i >> 1) * 8;



            /* WARNING: do no forget half pels */

            src_x = av_clip(src_x, -16, s->width);

            if (src_x == s->width)

                dxy &= ~3;

            src_y = av_clip(src_y, -16, s->height);

            if (src_y == s->height)

                dxy &= ~12;



            ptr = ref_picture[0] + (src_y * s->linesize) + (src_x);

            if ((unsigned)src_x > FFMAX(s->h_edge_pos - (motion_x & 3) - 8, 0) ||

                (unsigned)src_y > FFMAX(s->v_edge_pos - (motion_y & 3) - 8, 0)) {

                s->vdsp.emulated_edge_mc(s->edge_emu_buffer, ptr,

                                         s->linesize, s->linesize,

                                         9, 9,

                                         src_x, src_y,

                                         s->h_edge_pos,

                                         s->v_edge_pos);

                ptr = s->edge_emu_buffer;

            }

            dest = dest_y + ((i & 1) * 8) + (i >> 1) * 8 * s->linesize;

            qpix_op[1][dxy](dest, ptr, s->linesize);



            mx += s->mv[dir][i][0] / 2;

            my += s->mv[dir][i][1] / 2;

        }

    } else {

        for (i = 0; i < 4; i++) {

            hpel_motion(s,

                        dest_y + ((i & 1) * 8) + (i >> 1) * 8 * s->linesize,

                        ref_picture[0],

                        mb_x * 16 + (i & 1) * 8,

                        mb_y * 16 + (i >> 1) * 8,

                        pix_op[1],

                        s->mv[dir][i][0],

                        s->mv[dir][i][1]);



            mx += s->mv[dir][i][0];

            my += s->mv[dir][i][1];

        }

    }



    if (!CONFIG_GRAY || !(s->flags & CODEC_FLAG_GRAY))

        chroma_4mv_motion(s, dest_cb, dest_cr,

                          ref_picture, pix_op[1], mx, my);

}
