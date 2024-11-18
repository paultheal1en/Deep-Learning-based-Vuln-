static inline void xchg_mb_border(H264Context *h, uint8_t *src_y, uint8_t *src_cb, uint8_t *src_cr, int linesize, int uvlinesize, int xchg){

    MpegEncContext * const s = &h->s;

    int temp8, i;

    uint64_t temp64;



    src_y  -=   linesize + 1;

    src_cb -= uvlinesize + 1;

    src_cr -= uvlinesize + 1;



#define XCHG(a,b,t,xchg)\

t= a;\

if(xchg)\

    a= b;\

b= t;

    

    for(i=0; i<17; i++){

        XCHG(h->left_border[i     ], src_y [i*  linesize], temp8, xchg);

    }

    

    XCHG(*(uint64_t*)(h->top_border[s->mb_x]+0), *(uint64_t*)(src_y +1), temp64, xchg);

    XCHG(*(uint64_t*)(h->top_border[s->mb_x]+8), *(uint64_t*)(src_y +9), temp64, 1);



    if(!(s->flags&CODEC_FLAG_GRAY)){

        for(i=0; i<9; i++){

            XCHG(h->left_border[i+17  ], src_cb[i*uvlinesize], temp8, xchg);

            XCHG(h->left_border[i+17+9], src_cr[i*uvlinesize], temp8, xchg);

        }

        XCHG(*(uint64_t*)(h->top_border[s->mb_x]+16), *(uint64_t*)(src_cb+1), temp64, 1);

        XCHG(*(uint64_t*)(h->top_border[s->mb_x]+24), *(uint64_t*)(src_cr+1), temp64, 1);

    }

}
