static void filter(struct vf_priv_s *p, uint8_t *dst[3], uint8_t *src[3], int dst_stride[3], int src_stride[3], int width, int height){

    int x, y, i;



    for(i=0; i<3; i++){

        p->frame->data[i]= src[i];

        p->frame->linesize[i]= src_stride[i];

    }



    p->avctx_enc->me_cmp=

    p->avctx_enc->me_sub_cmp= FF_CMP_SAD /*| (p->parity ? FF_CMP_ODD : FF_CMP_EVEN)*/;

    p->frame->quality= p->qp*FF_QP2LAMBDA;

    avcodec_encode_video(p->avctx_enc, p->outbuf, p->outbuf_size, p->frame);

    p->frame_dec = p->avctx_enc->coded_frame;



    for(i=0; i<3; i++){

        int is_chroma= !!i;

        int w= width >>is_chroma;

        int h= height>>is_chroma;

        int fils= p->frame_dec->linesize[i];

        int srcs= src_stride[i];



        for(y=0; y<h; y++){

            if((y ^ p->parity) & 1){

                for(x=0; x<w; x++){

                    if((x-2)+(y-1)*w>=0 && (x+2)+(y+1)*w<w*h){ //FIXME either alloc larger images or optimize this

                        uint8_t *filp= &p->frame_dec->data[i][x + y*fils];

                        uint8_t *srcp= &src[i][x + y*srcs];

                        int diff0= filp[-fils] - srcp[-srcs];

                        int diff1= filp[+fils] - srcp[+srcs];

                        int spatial_score= ABS(srcp[-srcs-1] - srcp[+srcs-1])

                                          +ABS(srcp[-srcs  ] - srcp[+srcs  ])

                                          +ABS(srcp[-srcs+1] - srcp[+srcs+1]) - 1;

                        int temp= filp[0];



#define CHECK(j)\

    {   int score= ABS(srcp[-srcs-1+(j)] - srcp[+srcs-1-(j)])\

                 + ABS(srcp[-srcs  +(j)] - srcp[+srcs  -(j)])\

                 + ABS(srcp[-srcs+1+(j)] - srcp[+srcs+1-(j)]);\

        if(score < spatial_score){\

            spatial_score= score;\

            diff0= filp[-fils+(j)] - srcp[-srcs+(j)];\

            diff1= filp[+fils-(j)] - srcp[+srcs-(j)];



                        CHECK(-1) CHECK(-2) }} }}

                        CHECK( 1) CHECK( 2) }} }}
