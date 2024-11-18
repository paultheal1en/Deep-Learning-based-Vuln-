static void choose_pixel_fmt(AVStream *st, AVCodec *codec)

{

    if(codec && codec->pix_fmts){

        const enum PixelFormat *p= codec->pix_fmts;

        if(st->codec->strict_std_compliance <= FF_COMPLIANCE_UNOFFICIAL){

            if(st->codec->codec_id==CODEC_ID_MJPEG){

                p= (const enum PixelFormat[]){PIX_FMT_YUVJ420P, PIX_FMT_YUVJ422P, PIX_FMT_YUV420P, PIX_FMT_YUV422P, PIX_FMT_NONE};

            }else if(st->codec->codec_id==CODEC_ID_LJPEG){

                p= (const enum PixelFormat[]){PIX_FMT_YUVJ420P, PIX_FMT_YUVJ422P, PIX_FMT_YUVJ444P, PIX_FMT_YUV420P, PIX_FMT_YUV422P, PIX_FMT_YUV444P, PIX_FMT_BGRA, PIX_FMT_NONE};

            }

        }

        for(; *p!=-1; p++){

            if(*p == st->codec->pix_fmt)

                break;

        }

        if (*p == -1) {


            av_log(NULL, AV_LOG_WARNING,

                   "Incompatible pixel format '%s' for codec '%s', auto-selecting format '%s'\n",

                   av_pix_fmt_descriptors[st->codec->pix_fmt].name,

                   codec->name,

                   av_pix_fmt_descriptors[codec->pix_fmts[0]].name);

            st->codec->pix_fmt = codec->pix_fmts[0];

        }

    }

}