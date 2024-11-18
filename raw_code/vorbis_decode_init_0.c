static av_cold int vorbis_decode_init(AVCodecContext *avccontext)

{

    vorbis_context *vc = avccontext->priv_data;

    uint8_t *headers   = avccontext->extradata;

    int headers_len    = avccontext->extradata_size;

    uint8_t *header_start[3];

    int header_len[3];

    GetBitContext *gb = &vc->gb;

    int hdr_type, ret;



    vc->avccontext = avccontext;

    ff_dsputil_init(&vc->dsp, avccontext);

    ff_fmt_convert_init(&vc->fmt_conv, avccontext);



    if (avccontext->request_sample_fmt == AV_SAMPLE_FMT_FLT) {

        avccontext->sample_fmt = AV_SAMPLE_FMT_FLT;

        vc->scale_bias = 1.0f;

    } else {

        avccontext->sample_fmt = AV_SAMPLE_FMT_S16;

        vc->scale_bias = 32768.0f;

    }



    if (!headers_len) {

        av_log(avccontext, AV_LOG_ERROR, "Extradata missing.\n");

        return AVERROR_INVALIDDATA;

    }



    if ((ret = avpriv_split_xiph_headers(headers, headers_len, 30, header_start, header_len)) < 0) {

        av_log(avccontext, AV_LOG_ERROR, "Extradata corrupt.\n");

        return ret;

    }



    init_get_bits(gb, header_start[0], header_len[0]*8);

    hdr_type = get_bits(gb, 8);

    if (hdr_type != 1) {

        av_log(avccontext, AV_LOG_ERROR, "First header is not the id header.\n");

        return AVERROR_INVALIDDATA;

    }

    if ((ret = vorbis_parse_id_hdr(vc))) {

        av_log(avccontext, AV_LOG_ERROR, "Id header corrupt.\n");

        vorbis_free(vc);

        return ret;

    }



    init_get_bits(gb, header_start[2], header_len[2]*8);

    hdr_type = get_bits(gb, 8);

    if (hdr_type != 5) {

        av_log(avccontext, AV_LOG_ERROR, "Third header is not the setup header.\n");

        vorbis_free(vc);

        return AVERROR_INVALIDDATA;

    }

    if ((ret = vorbis_parse_setup_hdr(vc))) {

        av_log(avccontext, AV_LOG_ERROR, "Setup header corrupt.\n");

        vorbis_free(vc);

        return ret;

    }



    if (vc->audio_channels > 8)

        avccontext->channel_layout = 0;

    else

        avccontext->channel_layout = ff_vorbis_channel_layouts[vc->audio_channels - 1];



    avccontext->channels    = vc->audio_channels;

    avccontext->sample_rate = vc->audio_samplerate;

    avccontext->frame_size  = FFMIN(vc->blocksize[0], vc->blocksize[1]) >> 2;



    avcodec_get_frame_defaults(&vc->frame);

    avccontext->coded_frame = &vc->frame;



    return 0;

}
