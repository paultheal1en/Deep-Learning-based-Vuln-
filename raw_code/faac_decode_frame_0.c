static int faac_decode_frame(AVCodecContext *avctx,

                             void *data, int *data_size,

                             uint8_t *buf, int buf_size)

{

    FAACContext *s = (FAACContext *) avctx->priv_data;

#ifndef FAAD2_VERSION

    unsigned long bytesconsumed;

    short *sample_buffer = NULL;

    unsigned long samples;

    int out;

#else

    faacDecFrameInfo frame_info;

    void *out;

#endif

    if(buf_size == 0)

        return 0;

#ifndef FAAD2_VERSION

    out = s->faacDecDecode(s->faac_handle,

                           (unsigned char*)buf,

                           &bytesconsumed,

                           data,

                           &samples);

    samples *= s->sample_size;

    if (data_size)

        *data_size = samples;

    return (buf_size < (int)bytesconsumed)

        ? buf_size : (int)bytesconsumed;

#else



    if(!s->init){

        unsigned long srate;

        unsigned char channels;

        int r = s->faacDecInit(s->faac_handle, buf, buf_size, &srate, &channels);

        if(r < 0){

            av_log(avctx, AV_LOG_ERROR, "faac: codec init failed: %s\n",

                   s->faacDecGetErrorMessage(frame_info.error));

            return 0;

        }

        avctx->sample_rate = srate;

        avctx->channels = channels;

        s->init = 1;

    }



    out = s->faacDecDecode(s->faac_handle, &frame_info, (unsigned char*)buf, (unsigned long)buf_size);



    if (frame_info.error > 0) {

        av_log(avctx, AV_LOG_ERROR, "faac: frame decoding failed: %s\n",

                s->faacDecGetErrorMessage(frame_info.error));

        return 0;

    }



    frame_info.samples *= s->sample_size;

    memcpy(data, out, frame_info.samples); // CHECKME - can we cheat this one



    if (data_size)

        *data_size = frame_info.samples;



    return (buf_size < (int)frame_info.bytesconsumed)

        ? buf_size : (int)frame_info.bytesconsumed;

#endif

}
