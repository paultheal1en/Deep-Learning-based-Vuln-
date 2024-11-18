static inline void xchg_mb_border(H264Context *h, uint8_t *src_y, uint8_t *src_cb, uint8_t *src_cr, int linesize, int uvlinesize, int xchg, int simple){

    MpegEncContext * const s = &h->s;

    int temp8, i;

    uint64_t temp64;

    int deblock_left;

    int deblock_top;

    int mb_xy;

    int step    = 1;

    int offset  = 1;

    int uvoffset= 1;

    int top_idx = 1;



    if(!simple && FRAME_MBAFF){

        if(s->mb_y&1){

            offset  = MB_MBAFF ? 1 : 17;

            uvoffset= MB_MBAFF ? 1 : 9;

        }else{

            offset  =

            uvoffset=

            top_idx = MB_MBAFF ? 0 : 1;

        }

        step= MB_MBAFF ? 2 : 1;

    }



    if(h->deblocking_filter == 2) {

        mb_xy = h->mb_xy;

        deblock_left = h->slice_table[mb_xy] == h->slice_table[mb_xy - 1];

        deblock_top  = h->slice_table[mb_xy] == h->slice_table[h->top_mb_xy];

    } else {

        deblock_left = (s->mb_x > 0);

        deblock_top =  (s->mb_y > !!MB_FIELD);

    }



    src_y  -=   linesize + 1;

    src_cb -= uvlinesize + 1;

    src_cr -= uvlinesize + 1;



#define XCHG(a,b,t,xchg)\

t= a;\

if(xchg)\

    a= b;\

b= t;



    if(deblock_left){

        for(i = !deblock_top; i<16; i++){

            XCHG(h->left_border[offset+i*step], src_y [i*  linesize], temp8, xchg);

        }

        XCHG(h->left_border[offset+i*step], src_y [i*  linesize], temp8, 1);

    }



    if(deblock_top){

        XCHG(*(uint64_t*)(h->top_borders[top_idx][s->mb_x]+0), *(uint64_t*)(src_y +1), temp64, xchg);

        XCHG(*(uint64_t*)(h->top_borders[top_idx][s->mb_x]+8), *(uint64_t*)(src_y +9), temp64, 1);

        if(s->mb_x+1 < s->mb_width){

            XCHG(*(uint64_t*)(h->top_borders[top_idx][s->mb_x+1]), *(uint64_t*)(src_y +17), temp64, 1);

        }

    }



    if(simple || !CONFIG_GRAY || !(s->flags&CODEC_FLAG_GRAY)){

        if(deblock_left){

            for(i = !deblock_top; i<8; i++){

                XCHG(h->left_border[uvoffset+34   +i*step], src_cb[i*uvlinesize], temp8, xchg);

                XCHG(h->left_border[uvoffset+34+18+i*step], src_cr[i*uvlinesize], temp8, xchg);

            }

            XCHG(h->left_border[uvoffset+34   +i*step], src_cb[i*uvlinesize], temp8, 1);

            XCHG(h->left_border[uvoffset+34+18+i*step], src_cr[i*uvlinesize], temp8, 1);

        }

        if(deblock_top){

            XCHG(*(uint64_t*)(h->top_borders[top_idx][s->mb_x]+16), *(uint64_t*)(src_cb+1), temp64, 1);

            XCHG(*(uint64_t*)(h->top_borders[top_idx][s->mb_x]+24), *(uint64_t*)(src_cr+1), temp64, 1);

        }

    }

}
