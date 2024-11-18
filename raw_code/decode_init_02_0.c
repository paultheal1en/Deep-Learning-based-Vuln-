static int decode_init(AVCodecContext *avctx)

{

    HYuvContext *s = avctx->priv_data;

    int width, height;



    s->avctx= avctx;

    s->flags= avctx->flags;

        

    dsputil_init(&s->dsp, avctx);

    memset(s->vlc, 0, 3*sizeof(VLC));

    

    width= s->width= avctx->width;

    height= s->height= avctx->height;

    avctx->coded_frame= &s->picture;



s->bgr32=1;

    assert(width && height);

//if(avctx->extradata)

//  printf("extradata:%X, extradata_size:%d\n", *(uint32_t*)avctx->extradata, avctx->extradata_size);

    if(avctx->extradata_size){

        if((avctx->bits_per_sample&7) && avctx->bits_per_sample != 12)

            s->version=1; // do such files exist at all?

        else

            s->version=2;

    }else

        s->version=0;

    

    if(s->version==2){

        int method;



        method= ((uint8_t*)avctx->extradata)[0];

        s->decorrelate= method&64 ? 1 : 0;

        s->predictor= method&63;

        s->bitstream_bpp= ((uint8_t*)avctx->extradata)[1];

        if(s->bitstream_bpp==0) 

            s->bitstream_bpp= avctx->bits_per_sample&~7;

        s->context= ((uint8_t*)avctx->extradata)[2] & 0x40 ? 1 : 0;

            

        if(read_huffman_tables(s, ((uint8_t*)avctx->extradata)+4, avctx->extradata_size) < 0)

            return -1;

    }else{

        switch(avctx->bits_per_sample&7){

        case 1:

            s->predictor= LEFT;

            s->decorrelate= 0;

            break;

        case 2:

            s->predictor= LEFT;

            s->decorrelate= 1;

            break;

        case 3:

            s->predictor= PLANE;

            s->decorrelate= avctx->bits_per_sample >= 24;

            break;

        case 4:

            s->predictor= MEDIAN;

            s->decorrelate= 0;

            break;

        default:

            s->predictor= LEFT; //OLD

            s->decorrelate= 0;

            break;

        }

        s->bitstream_bpp= avctx->bits_per_sample & ~7;

        s->context= 0;

        

        if(read_old_huffman_tables(s) < 0)

            return -1;

    }

    

    if(((uint8_t*)avctx->extradata)[2] & 0x20)

	s->interlaced= ((uint8_t*)avctx->extradata)[2] & 0x10 ? 1 : 0;

    else

	s->interlaced= height > 288;

    

    switch(s->bitstream_bpp){

    case 12:

        avctx->pix_fmt = PIX_FMT_YUV420P;

        break;

    case 16:

        if(s->yuy2){

            avctx->pix_fmt = PIX_FMT_YUV422;

        }else{

            avctx->pix_fmt = PIX_FMT_YUV422P;

        }

        break;

    case 24:

    case 32:

        if(s->bgr32){

            avctx->pix_fmt = PIX_FMT_RGBA32;

        }else{

            avctx->pix_fmt = PIX_FMT_BGR24;

        }

        break;

    default:

        assert(0);

    }

    

//    av_log(NULL, AV_LOG_DEBUG, "pred:%d bpp:%d hbpp:%d il:%d\n", s->predictor, s->bitstream_bpp, avctx->bits_per_sample, s->interlaced);



    return 0;

}
