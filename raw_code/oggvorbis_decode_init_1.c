static int oggvorbis_decode_init(AVCodecContext *avccontext) {

    OggVorbisDecContext *context = avccontext->priv_data ;

    uint8_t *p= avccontext->extradata;

    int i, hsizes[3];

    unsigned char *headers[3], *extradata = avccontext->extradata;



    vorbis_info_init(&context->vi) ;

    vorbis_comment_init(&context->vc) ;



    if(! avccontext->extradata_size || ! p) {

        av_log(avccontext, AV_LOG_ERROR, "vorbis extradata absent\n");

        return -1;

    }



    if(p[0] == 0 && p[1] == 30) {

        for(i = 0; i < 3; i++){

            hsizes[i] = bytestream_get_be16((const uint8_t **)&p);

            headers[i] = p;

            p += hsizes[i];

        }

    } else if(*p == 2) {

        unsigned int offset = 1;

        p++;

        for(i=0; i<2; i++) {

            hsizes[i] = 0;

            while((*p == 0xFF) && (offset < avccontext->extradata_size)) {

                hsizes[i] += 0xFF;

                offset++;

                p++;

            }

            if(offset >= avccontext->extradata_size - 1) {

                av_log(avccontext, AV_LOG_ERROR,

                       "vorbis header sizes damaged\n");

                return -1;

            }

            hsizes[i] += *p;

            offset++;

            p++;

        }

        hsizes[2] = avccontext->extradata_size - hsizes[0]-hsizes[1]-offset;

#if 0

        av_log(avccontext, AV_LOG_DEBUG,

               "vorbis header sizes: %d, %d, %d, / extradata_len is %d \n",

               hsizes[0], hsizes[1], hsizes[2], avccontext->extradata_size);

#endif

        headers[0] = extradata + offset;

        headers[1] = extradata + offset + hsizes[0];

        headers[2] = extradata + offset + hsizes[0] + hsizes[1];

    } else {

        av_log(avccontext, AV_LOG_ERROR,

               "vorbis initial header len is wrong: %d\n", *p);

        return -1;

    }



    for(i=0; i<3; i++){

        context->op.b_o_s= i==0;

        context->op.bytes = hsizes[i];

        context->op.packet = headers[i];

        if(vorbis_synthesis_headerin(&context->vi, &context->vc, &context->op)<0){

            av_log(avccontext, AV_LOG_ERROR, "%d. vorbis header damaged\n", i+1);

            return -1;

        }

    }



    avccontext->channels = context->vi.channels;

    avccontext->sample_rate = context->vi.rate;

    avccontext->sample_fmt = AV_SAMPLE_FMT_S16;

    avccontext->time_base= (AVRational){1, avccontext->sample_rate};



    vorbis_synthesis_init(&context->vd, &context->vi);

    vorbis_block_init(&context->vd, &context->vb);



    return 0 ;

}
