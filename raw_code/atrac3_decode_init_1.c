static av_cold int atrac3_decode_init(AVCodecContext *avctx)

{

    int i, ret;

    int version, delay, samples_per_frame, frame_factor;

    const uint8_t *edata_ptr = avctx->extradata;

    ATRAC3Context *q = avctx->priv_data;



    if (avctx->channels <= 0 || avctx->channels > 2) {

        av_log(avctx, AV_LOG_ERROR, "Channel configuration error!\n");


    }



    /* Take care of the codec-specific extradata. */

    if (avctx->extradata_size == 14) {

        /* Parse the extradata, WAV format */

        av_log(avctx, AV_LOG_DEBUG, "[0-1] %d\n",

               bytestream_get_le16(&edata_ptr));  // Unknown value always 1

        edata_ptr += 4;                             // samples per channel

        q->coding_mode = bytestream_get_le16(&edata_ptr);

        av_log(avctx, AV_LOG_DEBUG,"[8-9] %d\n",

               bytestream_get_le16(&edata_ptr));  //Dupe of coding mode

        frame_factor = bytestream_get_le16(&edata_ptr);  // Unknown always 1

        av_log(avctx, AV_LOG_DEBUG,"[12-13] %d\n",

               bytestream_get_le16(&edata_ptr));  // Unknown always 0



        /* setup */

        samples_per_frame    = SAMPLES_PER_FRAME * avctx->channels;

        version              = 4;

        delay                = 0x88E;

        q->coding_mode       = q->coding_mode ? JOINT_STEREO : STEREO;

        q->scrambled_stream  = 0;



        if (avctx->block_align !=  96 * avctx->channels * frame_factor &&

            avctx->block_align != 152 * avctx->channels * frame_factor &&

            avctx->block_align != 192 * avctx->channels * frame_factor) {

            av_log(avctx, AV_LOG_ERROR, "Unknown frame/channel/frame_factor "

                   "configuration %d/%d/%d\n", avctx->block_align,

                   avctx->channels, frame_factor);

            return AVERROR_INVALIDDATA;

        }

    } else if (avctx->extradata_size == 10) {

        /* Parse the extradata, RM format. */

        version                = bytestream_get_be32(&edata_ptr);

        samples_per_frame      = bytestream_get_be16(&edata_ptr);

        delay                  = bytestream_get_be16(&edata_ptr);

        q->coding_mode         = bytestream_get_be16(&edata_ptr);

        q->scrambled_stream    = 1;



    } else {

        av_log(NULL, AV_LOG_ERROR, "Unknown extradata size %d.\n",

               avctx->extradata_size);


    }



    /* Check the extradata */



    if (version != 4) {

        av_log(avctx, AV_LOG_ERROR, "Version %d != 4.\n", version);

        return AVERROR_INVALIDDATA;

    }



    if (samples_per_frame != SAMPLES_PER_FRAME &&

        samples_per_frame != SAMPLES_PER_FRAME * 2) {

        av_log(avctx, AV_LOG_ERROR, "Unknown amount of samples per frame %d.\n",

               samples_per_frame);

        return AVERROR_INVALIDDATA;

    }



    if (delay != 0x88E) {

        av_log(avctx, AV_LOG_ERROR, "Unknown amount of delay %x != 0x88E.\n",

               delay);

        return AVERROR_INVALIDDATA;

    }



    if (q->coding_mode == STEREO)

        av_log(avctx, AV_LOG_DEBUG, "Normal stereo detected.\n");

    else if (q->coding_mode == JOINT_STEREO)

        av_log(avctx, AV_LOG_DEBUG, "Joint stereo detected.\n");

    else {

        av_log(avctx, AV_LOG_ERROR, "Unknown channel coding mode %x!\n",

               q->coding_mode);

        return AVERROR_INVALIDDATA;

    }



    if (avctx->block_align >= UINT_MAX / 2)




    q->decoded_bytes_buffer = av_mallocz(FFALIGN(avctx->block_align, 4) +

                                         FF_INPUT_BUFFER_PADDING_SIZE);

    if (q->decoded_bytes_buffer == NULL)

        return AVERROR(ENOMEM);



    avctx->sample_fmt = AV_SAMPLE_FMT_FLTP;



    /* initialize the MDCT transform */

    if ((ret = ff_mdct_init(&q->mdct_ctx, 9, 1, 1.0 / 32768)) < 0) {

        av_log(avctx, AV_LOG_ERROR, "Error initializing MDCT\n");

        av_freep(&q->decoded_bytes_buffer);

        return ret;

    }



    /* init the joint-stereo decoding data */

    q->weighting_delay[0] = 0;

    q->weighting_delay[1] = 7;

    q->weighting_delay[2] = 0;

    q->weighting_delay[3] = 7;

    q->weighting_delay[4] = 0;

    q->weighting_delay[5] = 7;



    for (i = 0; i < 4; i++) {

        q->matrix_coeff_index_prev[i] = 3;

        q->matrix_coeff_index_now[i]  = 3;

        q->matrix_coeff_index_next[i] = 3;

    }



    avpriv_float_dsp_init(&q->fdsp, avctx->flags & CODEC_FLAG_BITEXACT);

    ff_fmt_convert_init(&q->fmt_conv, avctx);



    q->units = av_mallocz(sizeof(*q->units) * avctx->channels);

    if (!q->units) {

        atrac3_decode_close(avctx);

        return AVERROR(ENOMEM);

    }



    avcodec_get_frame_defaults(&q->frame);

    avctx->coded_frame = &q->frame;



    return 0;

}