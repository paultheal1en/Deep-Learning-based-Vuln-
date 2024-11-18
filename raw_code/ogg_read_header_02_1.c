static int ogg_read_header(AVFormatContext *avfcontext, AVFormatParameters *ap)

{

    OggContext *context = avfcontext->priv_data;

    ogg_packet op ;    

    char *buf ;

    ogg_page og ;

    AVStream *ast ;

    AVCodecContext *codec;

    uint8_t *p;

    int i;

     

    ogg_sync_init(&context->oy) ;

    buf = ogg_sync_buffer(&context->oy, DECODER_BUFFER_SIZE) ;



    if(get_buffer(&avfcontext->pb, buf, DECODER_BUFFER_SIZE) <= 0)

	return AVERROR_IO ;

    

    ogg_sync_wrote(&context->oy, DECODER_BUFFER_SIZE) ;   

    ogg_sync_pageout(&context->oy, &og) ;

    ogg_stream_init(&context->os, ogg_page_serialno(&og)) ;

    ogg_stream_pagein(&context->os, &og) ;

    

    /* currently only one vorbis stream supported */



    ast = av_new_stream(avfcontext, 0) ;

    if(!ast)

	return AVERROR_NOMEM ;

    av_set_pts_info(ast, 60, 1, AV_TIME_BASE);



    codec= &ast->codec;

    codec->codec_type = CODEC_TYPE_AUDIO;

    codec->codec_id = CODEC_ID_VORBIS;

    for(i=0; i<3; i++){

        if(next_packet(avfcontext, &op)){


        }



        codec->extradata_size+= 2 + op.bytes;

        codec->extradata= av_realloc(codec->extradata, codec->extradata_size + FF_INPUT_BUFFER_PADDING_SIZE);

        p= codec->extradata + codec->extradata_size - 2 - op.bytes;

        *(p++)= op.bytes>>8;

        *(p++)= op.bytes&0xFF;

        memcpy(p, op.packet, op.bytes);

    }



    return 0 ;

}