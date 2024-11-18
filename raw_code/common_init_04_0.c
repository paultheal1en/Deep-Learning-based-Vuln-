static int common_init(AVCodecContext *avctx){

    SnowContext *s = avctx->priv_data;

    int width, height;

    int level, orientation, plane_index, dec;



    s->avctx= avctx;

        

    dsputil_init(&s->dsp, avctx);



#define mcf(dx,dy)\

    s->dsp.put_qpel_pixels_tab       [0][dy+dx/4]=\

    s->dsp.put_no_rnd_qpel_pixels_tab[0][dy+dx/4]=\

        mc_block ## dx ## dy;



    mcf( 0, 0)

    mcf( 4, 0)

    mcf( 8, 0)

    mcf(12, 0)

    mcf( 0, 4)

    mcf( 4, 4)

    mcf( 8, 4)

    mcf(12, 4)

    mcf( 0, 8)

    mcf( 4, 8)

    mcf( 8, 8)

    mcf(12, 8)

    mcf( 0,12)

    mcf( 4,12)

    mcf( 8,12)

    mcf(12,12)



#define mcfh(dx,dy)\

    s->dsp.put_pixels_tab       [0][dy/4+dx/8]=\

    s->dsp.put_no_rnd_pixels_tab[0][dy/4+dx/8]=\

        mc_block_hpel ## dx ## dy;



    mcfh(0, 0)

    mcfh(8, 0)

    mcfh(0, 8)

    mcfh(8, 8)

        

    dec= s->spatial_decomposition_count= 5;

    s->spatial_decomposition_type= avctx->prediction_method; //FIXME add decorrelator type r transform_type

    

    s->chroma_h_shift= 1; //FIXME XXX

    s->chroma_v_shift= 1;

    

//    dec += FFMAX(s->chroma_h_shift, s->chroma_v_shift);

    

    s->b_width = (s->avctx->width +(1<<dec)-1)>>dec;

    s->b_height= (s->avctx->height+(1<<dec)-1)>>dec;

    

    s->spatial_dwt_buffer= av_mallocz(s->b_width*s->b_height*sizeof(DWTELEM)<<(2*dec));

    s->pred_buffer= av_mallocz(s->b_width*s->b_height*sizeof(DWTELEM)<<(2*dec));

    

    s->mv_scale= (s->avctx->flags & CODEC_FLAG_QPEL) ? 2 : 4;

    

    for(plane_index=0; plane_index<3; plane_index++){    

        int w= s->avctx->width;

        int h= s->avctx->height;



        if(plane_index){

            w>>= s->chroma_h_shift;

            h>>= s->chroma_v_shift;

        }

        s->plane[plane_index].width = w;

        s->plane[plane_index].height= h;

//av_log(NULL, AV_LOG_DEBUG, "%d %d\n", w, h);

        for(level=s->spatial_decomposition_count-1; level>=0; level--){

            for(orientation=level ? 1 : 0; orientation<4; orientation++){

                SubBand *b= &s->plane[plane_index].band[level][orientation];

                

                b->buf= s->spatial_dwt_buffer;

                b->level= level;

                b->stride= s->plane[plane_index].width << (s->spatial_decomposition_count - level);

                b->width = (w + !(orientation&1))>>1;

                b->height= (h + !(orientation>1))>>1;

                

                if(orientation&1) b->buf += (w+1)>>1;

                if(orientation>1) b->buf += b->stride>>1;

                

                if(level)

                    b->parent= &s->plane[plane_index].band[level-1][orientation];

            }

            w= (w+1)>>1;

            h= (h+1)>>1;

        }

    }

    

    //FIXME init_subband() ?

    s->mb_band.stride= s->mv_band[0].stride= s->mv_band[1].stride=

    s->mb_band.width = s->mv_band[0].width = s->mv_band[1].width = (s->avctx->width + 15)>>4;

    s->mb_band.height= s->mv_band[0].height= s->mv_band[1].height= (s->avctx->height+ 15)>>4;

    s->mb_band   .buf= av_mallocz(s->mb_band   .stride * s->mb_band   .height*sizeof(DWTELEM));

    s->mv_band[0].buf= av_mallocz(s->mv_band[0].stride * s->mv_band[0].height*sizeof(DWTELEM));

    s->mv_band[1].buf= av_mallocz(s->mv_band[1].stride * s->mv_band[1].height*sizeof(DWTELEM));



    reset_contexts(s);

/*    

    width= s->width= avctx->width;

    height= s->height= avctx->height;

    

    assert(width && height);

*/

    s->avctx->get_buffer(s->avctx, &s->mconly_picture);

    

    return 0;

}
