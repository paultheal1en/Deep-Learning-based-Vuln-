static inline void mc_dir_part(H264Context *h, Picture *pic, int n, int square, int chroma_height, int delta, int list,

                           uint8_t *dest_y, uint8_t *dest_cb, uint8_t *dest_cr,

                           int src_x_offset, int src_y_offset,

                           qpel_mc_func *qpix_op, h264_chroma_mc_func chroma_op){

    MpegEncContext * const s = &h->s;

    const int mx= h->mv_cache[list][ scan8[n] ][0] + src_x_offset*8;

    int my=       h->mv_cache[list][ scan8[n] ][1] + src_y_offset*8;

    const int luma_xy= (mx&3) + ((my&3)<<2);

    uint8_t * src_y = pic->data[0] + (mx>>2) + (my>>2)*h->mb_linesize;

    uint8_t * src_cb, * src_cr;

    int extra_width= h->emu_edge_width;

    int extra_height= h->emu_edge_height;

    int emu=0;

    const int full_mx= mx>>2;

    const int full_my= my>>2;

    const int pic_width  = 16*s->mb_width;

    const int pic_height = 16*s->mb_height >> MB_FIELD;



    if(!pic->data[0]) //FIXME this is unacceptable, some sensible error concealment must be done for missing reference frames

        return;



    if(mx&7) extra_width -= 3;

    if(my&7) extra_height -= 3;



    if(   full_mx < 0-extra_width

       || full_my < 0-extra_height

       || full_mx + 16/*FIXME*/ > pic_width + extra_width

       || full_my + 16/*FIXME*/ > pic_height + extra_height){

        ff_emulated_edge_mc(s->edge_emu_buffer, src_y - 2 - 2*h->mb_linesize, h->mb_linesize, 16+5, 16+5/*FIXME*/, full_mx-2, full_my-2, pic_width, pic_height);

            src_y= s->edge_emu_buffer + 2 + 2*h->mb_linesize;

        emu=1;

    }



    qpix_op[luma_xy](dest_y, src_y, h->mb_linesize); //FIXME try variable height perhaps?

    if(!square){

        qpix_op[luma_xy](dest_y + delta, src_y + delta, h->mb_linesize);

    }



    if(ENABLE_GRAY && s->flags&CODEC_FLAG_GRAY) return;



    if(MB_FIELD){

        // chroma offset when predicting from a field of opposite parity

        my += 2 * ((s->mb_y & 1) - (pic->reference - 1));

        emu |= (my>>3) < 0 || (my>>3) + 8 >= (pic_height>>1);

    }

    src_cb= pic->data[1] + (mx>>3) + (my>>3)*h->mb_uvlinesize;

    src_cr= pic->data[2] + (mx>>3) + (my>>3)*h->mb_uvlinesize;



    if(emu){

        ff_emulated_edge_mc(s->edge_emu_buffer, src_cb, h->mb_uvlinesize, 9, 9/*FIXME*/, (mx>>3), (my>>3), pic_width>>1, pic_height>>1);

            src_cb= s->edge_emu_buffer;

    }

    chroma_op(dest_cb, src_cb, h->mb_uvlinesize, chroma_height, mx&7, my&7);



    if(emu){

        ff_emulated_edge_mc(s->edge_emu_buffer, src_cr, h->mb_uvlinesize, 9, 9/*FIXME*/, (mx>>3), (my>>3), pic_width>>1, pic_height>>1);

            src_cr= s->edge_emu_buffer;

    }

    chroma_op(dest_cr, src_cr, h->mb_uvlinesize, chroma_height, mx&7, my&7);

}
