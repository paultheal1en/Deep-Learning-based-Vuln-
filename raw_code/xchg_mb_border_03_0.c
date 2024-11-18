static inline void xchg_mb_border(H264Context *h, uint8_t *src_y,

                                  uint8_t *src_cb, uint8_t *src_cr,

                                  int linesize, int uvlinesize,

                                  int xchg, int simple, int pixel_shift){

    MpegEncContext * const s = &h->s;

    int deblock_left;

    int deblock_top;

    int top_idx = 1;

    uint8_t *top_border_m1;

    uint8_t *top_border;



    if(!simple && FRAME_MBAFF){

        if(s->mb_y&1){

            if(!MB_MBAFF)

                return;

        }else{

            top_idx = MB_MBAFF ? 0 : 1;

        }

    }



    if(h->deblocking_filter == 2) {

        deblock_left = h->left_type[0];

        deblock_top  = h->top_type;

    } else {

        deblock_left = (s->mb_x > 0);

        deblock_top =  (s->mb_y > !!MB_FIELD);

    }



    src_y  -=   linesize + 1 + pixel_shift;

    src_cb -= uvlinesize + 1 + pixel_shift;

    src_cr -= uvlinesize + 1 + pixel_shift;



    top_border_m1 = h->top_borders[top_idx][s->mb_x-1];

    top_border    = h->top_borders[top_idx][s->mb_x];



#define XCHG(a,b,xchg)\

    if (pixel_shift) {\

        if (xchg) {\

            AV_SWAP64(b+0,a+0);\

            AV_SWAP64(b+8,a+8);\

        } else {\

            AV_COPY128(b,a); \

        }\

    } else \

if (xchg) AV_SWAP64(b,a);\

else      AV_COPY64(b,a);



    if(deblock_top){

        if(deblock_left){

            XCHG(top_border_m1 + (8 << pixel_shift), src_y - (7 << pixel_shift), 1);

        }

        XCHG(top_border + (0 << pixel_shift), src_y + (1 << pixel_shift), xchg);

        XCHG(top_border + (8 << pixel_shift), src_y + (9 << pixel_shift), 1);

        if(s->mb_x+1 < s->mb_width){

            XCHG(h->top_borders[top_idx][s->mb_x+1], src_y + (17 << pixel_shift), 1);

        }

    }

    if(simple || !CONFIG_GRAY || !(s->flags&CODEC_FLAG_GRAY)){

        if(deblock_top){

            if(deblock_left){

                XCHG(top_border_m1 + (16 << pixel_shift), src_cb - (7 << pixel_shift), 1);

                XCHG(top_border_m1 + (24 << pixel_shift), src_cr - (7 << pixel_shift), 1);

            }

            XCHG(top_border + (16 << pixel_shift), src_cb+1+pixel_shift, 1);

            XCHG(top_border + (24 << pixel_shift), src_cr+1+pixel_shift, 1);

        }

    }

}
