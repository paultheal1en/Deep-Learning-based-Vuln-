static inline void rv34_mc(RV34DecContext *r, const int block_type,

                          const int xoff, const int yoff, int mv_off,

                          const int width, const int height, int dir,

                          const int thirdpel, int weighted,

                          qpel_mc_func (*qpel_mc)[16],

                          h264_chroma_mc_func (*chroma_mc))

{

    MpegEncContext *s = &r->s;

    uint8_t *Y, *U, *V, *srcY, *srcU, *srcV;

    int dxy, mx, my, umx, umy, lx, ly, uvmx, uvmy, src_x, src_y, uvsrc_x, uvsrc_y;

    int mv_pos = s->mb_x * 2 + s->mb_y * 2 * s->b8_stride + mv_off;

    int is16x16 = 1;



    if(thirdpel){

        int chroma_mx, chroma_my;

        mx = (s->current_picture_ptr->f.motion_val[dir][mv_pos][0] + (3 << 24)) / 3 - (1 << 24);

        my = (s->current_picture_ptr->f.motion_val[dir][mv_pos][1] + (3 << 24)) / 3 - (1 << 24);

        lx = (s->current_picture_ptr->f.motion_val[dir][mv_pos][0] + (3 << 24)) % 3;

        ly = (s->current_picture_ptr->f.motion_val[dir][mv_pos][1] + (3 << 24)) % 3;

        chroma_mx = s->current_picture_ptr->f.motion_val[dir][mv_pos][0] / 2;

        chroma_my = s->current_picture_ptr->f.motion_val[dir][mv_pos][1] / 2;

        umx = (chroma_mx + (3 << 24)) / 3 - (1 << 24);

        umy = (chroma_my + (3 << 24)) / 3 - (1 << 24);

        uvmx = chroma_coeffs[(chroma_mx + (3 << 24)) % 3];

        uvmy = chroma_coeffs[(chroma_my + (3 << 24)) % 3];

    }else{

        int cx, cy;

        mx = s->current_picture_ptr->f.motion_val[dir][mv_pos][0] >> 2;

        my = s->current_picture_ptr->f.motion_val[dir][mv_pos][1] >> 2;

        lx = s->current_picture_ptr->f.motion_val[dir][mv_pos][0] & 3;

        ly = s->current_picture_ptr->f.motion_val[dir][mv_pos][1] & 3;

        cx = s->current_picture_ptr->f.motion_val[dir][mv_pos][0] / 2;

        cy = s->current_picture_ptr->f.motion_val[dir][mv_pos][1] / 2;

        umx = cx >> 2;

        umy = cy >> 2;

        uvmx = (cx & 3) << 1;

        uvmy = (cy & 3) << 1;

        //due to some flaw RV40 uses the same MC compensation routine for H2V2 and H3V3

        if(uvmx == 6 && uvmy == 6)

            uvmx = uvmy = 4;

    }

    dxy = ly*4 + lx;

    srcY = dir ? s->next_picture_ptr->f.data[0] : s->last_picture_ptr->f.data[0];

    srcU = dir ? s->next_picture_ptr->f.data[1] : s->last_picture_ptr->f.data[1];

    srcV = dir ? s->next_picture_ptr->f.data[2] : s->last_picture_ptr->f.data[2];

    src_x = s->mb_x * 16 + xoff + mx;

    src_y = s->mb_y * 16 + yoff + my;

    uvsrc_x = s->mb_x * 8 + (xoff >> 1) + umx;

    uvsrc_y = s->mb_y * 8 + (yoff >> 1) + umy;

    srcY += src_y * s->linesize + src_x;

    srcU += uvsrc_y * s->uvlinesize + uvsrc_x;

    srcV += uvsrc_y * s->uvlinesize + uvsrc_x;

    if(   (unsigned)(src_x - !!lx*2) > s->h_edge_pos - !!lx*2 - (width <<3) - 4

       || (unsigned)(src_y - !!ly*2) > s->v_edge_pos - !!ly*2 - (height<<3) - 4){

        uint8_t *uvbuf = s->edge_emu_buffer + 22 * s->linesize;



        srcY -= 2 + 2*s->linesize;

        s->dsp.emulated_edge_mc(s->edge_emu_buffer, srcY, s->linesize, (width<<3)+6, (height<<3)+6,

                            src_x - 2, src_y - 2, s->h_edge_pos, s->v_edge_pos);

        srcY = s->edge_emu_buffer + 2 + 2*s->linesize;

        s->dsp.emulated_edge_mc(uvbuf     , srcU, s->uvlinesize, (width<<2)+1, (height<<2)+1,

                            uvsrc_x, uvsrc_y, s->h_edge_pos >> 1, s->v_edge_pos >> 1);

        s->dsp.emulated_edge_mc(uvbuf + 16, srcV, s->uvlinesize, (width<<2)+1, (height<<2)+1,

                            uvsrc_x, uvsrc_y, s->h_edge_pos >> 1, s->v_edge_pos >> 1);

        srcU = uvbuf;

        srcV = uvbuf + 16;

    }

    if(!weighted){

        Y = s->dest[0] + xoff      + yoff     *s->linesize;

        U = s->dest[1] + (xoff>>1) + (yoff>>1)*s->uvlinesize;

        V = s->dest[2] + (xoff>>1) + (yoff>>1)*s->uvlinesize;

    }else{

        Y = r->tmp_b_block_y [dir]     +  xoff     +  yoff    *s->linesize;

        U = r->tmp_b_block_uv[dir*2]   + (xoff>>1) + (yoff>>1)*s->uvlinesize;

        V = r->tmp_b_block_uv[dir*2+1] + (xoff>>1) + (yoff>>1)*s->uvlinesize;

    }



    if(block_type == RV34_MB_P_16x8){

        qpel_mc[1][dxy](Y, srcY, s->linesize);

        Y    += 8;

        srcY += 8;

    }else if(block_type == RV34_MB_P_8x16){

        qpel_mc[1][dxy](Y, srcY, s->linesize);

        Y    += 8 * s->linesize;

        srcY += 8 * s->linesize;

    }

    is16x16 = (block_type != RV34_MB_P_8x8) && (block_type != RV34_MB_P_16x8) && (block_type != RV34_MB_P_8x16);

    qpel_mc[!is16x16][dxy](Y, srcY, s->linesize);

    chroma_mc[2-width]   (U, srcU, s->uvlinesize, height*4, uvmx, uvmy);

    chroma_mc[2-width]   (V, srcV, s->uvlinesize, height*4, uvmx, uvmy);

}
