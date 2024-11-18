void avcodec_align_dimensions(AVCodecContext *s, int *width, int *height){

    int w_align= 1;

    int h_align= 1;



    switch(s->pix_fmt){

    case PIX_FMT_YUV420P:

    case PIX_FMT_YUYV422:

    case PIX_FMT_UYVY422:

    case PIX_FMT_YUV422P:

    case PIX_FMT_YUV444P:

    case PIX_FMT_GRAY8:

    case PIX_FMT_GRAY16BE:

    case PIX_FMT_GRAY16LE:

    case PIX_FMT_YUVJ420P:

    case PIX_FMT_YUVJ422P:

    case PIX_FMT_YUVJ444P:

    case PIX_FMT_YUVA420P:

        w_align= 16; //FIXME check for non mpeg style codecs and use less alignment

        h_align= 16;



        break;

    case PIX_FMT_YUV411P:

    case PIX_FMT_UYYVYY411:

        w_align=32;

        h_align=8;

        break;

    case PIX_FMT_YUV410P:

        if(s->codec_id == CODEC_ID_SVQ1){

            w_align=64;

            h_align=64;

        }

    case PIX_FMT_RGB555:

        if(s->codec_id == CODEC_ID_RPZA){

            w_align=4;

            h_align=4;

        }

    case PIX_FMT_PAL8:

    case PIX_FMT_BGR8:

    case PIX_FMT_RGB8:

        if(s->codec_id == CODEC_ID_SMC){

            w_align=4;

            h_align=4;

        }

        break;

    case PIX_FMT_BGR24:

        if((s->codec_id == CODEC_ID_MSZH) || (s->codec_id == CODEC_ID_ZLIB)){

            w_align=4;

            h_align=4;

        }

        break;

    default:

        w_align= 1;

        h_align= 1;

        break;

    }



    *width = ALIGN(*width , w_align);

    *height= ALIGN(*height, h_align);

    if(s->codec_id == CODEC_ID_H264)

        *height+=2; // some of the optimized chroma MC reads one line too much

}