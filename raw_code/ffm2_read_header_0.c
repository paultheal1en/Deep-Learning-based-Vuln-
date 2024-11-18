static int ffm2_read_header(AVFormatContext *s)

{

    FFMContext *ffm = s->priv_data;

    AVStream *st;

    AVIOContext *pb = s->pb;

    AVCodecContext *codec, *dummy_codec = NULL;

    AVCodecParameters *codecpar;

    const AVCodecDescriptor *codec_desc;

    int ret;

    int f_main = 0, f_cprv = -1, f_stvi = -1, f_stau = -1;

    AVCodec *enc;

    char *buffer;



    ffm->packet_size = avio_rb32(pb);

    if (ffm->packet_size != FFM_PACKET_SIZE) {

        av_log(s, AV_LOG_ERROR, "Invalid packet size %d, expected size was %d\n",

               ffm->packet_size, FFM_PACKET_SIZE);

        ret = AVERROR_INVALIDDATA;

        goto fail;

    }



    ffm->write_index = avio_rb64(pb);

    /* get also filesize */

    if (pb->seekable) {

        ffm->file_size = avio_size(pb);

        if (ffm->write_index && 0)

            adjust_write_index(s);

    } else {

        ffm->file_size = (UINT64_C(1) << 63) - 1;

    }

    dummy_codec = avcodec_alloc_context3(NULL);



    while(!avio_feof(pb)) {

        unsigned id = avio_rb32(pb);

        unsigned size = avio_rb32(pb);

        int64_t next = avio_tell(pb) + size;

        char rc_eq_buf[128];



        if(!id)

            break;



        switch(id) {

        case MKBETAG('M', 'A', 'I', 'N'):

            if (f_main++) {

                ret = AVERROR(EINVAL);

                goto fail;

            }

            avio_rb32(pb); /* nb_streams */

            avio_rb32(pb); /* total bitrate */

            break;

        case MKBETAG('C', 'O', 'M', 'M'):

            f_cprv = f_stvi = f_stau = 0;

            st = avformat_new_stream(s, NULL);

            if (!st) {

                ret = AVERROR(ENOMEM);

                goto fail;

            }



            avpriv_set_pts_info(st, 64, 1, 1000000);



            codec = st->codec;

            codecpar = st->codecpar;

            /* generic info */

            codecpar->codec_id = avio_rb32(pb);

            codec_desc = avcodec_descriptor_get(codecpar->codec_id);

            if (!codec_desc) {

                av_log(s, AV_LOG_ERROR, "Invalid codec id: %d\n", codecpar->codec_id);

                codecpar->codec_id = AV_CODEC_ID_NONE;

                ret = AVERROR_INVALIDDATA;

                goto fail;

            }

            codecpar->codec_type = avio_r8(pb);

            if (codecpar->codec_type != codec_desc->type) {

                av_log(s, AV_LOG_ERROR, "Codec type mismatch: expected %d, found %d\n",

                       codec_desc->type, codecpar->codec_type);

                codecpar->codec_id = AV_CODEC_ID_NONE;

                codecpar->codec_type = AVMEDIA_TYPE_UNKNOWN;

                ret = AVERROR_INVALIDDATA;

                goto fail;

            }

            codecpar->bit_rate = avio_rb32(pb);

            if (codecpar->bit_rate < 0) {

                av_log(codec, AV_LOG_ERROR, "Invalid bit rate %"PRId64"\n", codecpar->bit_rate);

                ret = AVERROR_INVALIDDATA;

                goto fail;

            }

            codec->flags = avio_rb32(pb);

            codec->flags2 = avio_rb32(pb);

            codec->debug = avio_rb32(pb);

            if (codec->flags & AV_CODEC_FLAG_GLOBAL_HEADER) {

                int size = avio_rb32(pb);

                if (size < 0 || size >= FF_MAX_EXTRADATA_SIZE) {

                    av_log(s, AV_LOG_ERROR, "Invalid extradata size %d\n", size);

                    ret = AVERROR_INVALIDDATA;

                    goto fail;

                }

                codecpar->extradata = av_mallocz(size + AV_INPUT_BUFFER_PADDING_SIZE);

                if (!codecpar->extradata)

                    return AVERROR(ENOMEM);

                codecpar->extradata_size = size;

                avio_read(pb, codecpar->extradata, size);

            }

            break;

        case MKBETAG('S', 'T', 'V', 'I'):

            if (f_stvi++) {

                ret = AVERROR(EINVAL);

                goto fail;

            }

            codec->time_base.num = avio_rb32(pb);

            codec->time_base.den = avio_rb32(pb);

            if (codec->time_base.num <= 0 || codec->time_base.den <= 0) {

                av_log(s, AV_LOG_ERROR, "Invalid time base %d/%d\n",

                       codec->time_base.num, codec->time_base.den);

                ret = AVERROR_INVALIDDATA;

                goto fail;

            }

            codecpar->width = avio_rb16(pb);

            codecpar->height = avio_rb16(pb);

            ret = av_image_check_size(codecpar->width, codecpar->height, 0, s);

            if (ret < 0)

                goto fail;

            avio_rb16(pb); // gop_size

            codecpar->format = avio_rb32(pb);

            if (!av_pix_fmt_desc_get(codecpar->format)) {

                av_log(s, AV_LOG_ERROR, "Invalid pix fmt id: %d\n", codecpar->format);

                codecpar->format = AV_PIX_FMT_NONE;

                goto fail;

            }

            avio_r8(pb);   // qmin

            avio_r8(pb);   // qmax

            avio_r8(pb);   // max_qdiff

            avio_rb16(pb); // qcompress / 10000.0

            avio_rb16(pb); // qblur / 10000.0

            avio_rb32(pb); // bit_rate_tolerance

            avio_get_str(pb, INT_MAX, rc_eq_buf, sizeof(rc_eq_buf));



            avio_rb32(pb); // rc_max_rate

            avio_rb32(pb); // rc_min_rate

            avio_rb32(pb); // rc_buffer_size

            avio_rb64(pb); // i_quant_factor

            avio_rb64(pb); // b_quant_factor

            avio_rb64(pb); // i_quant_offset

            avio_rb64(pb); // b_quant_offset

            avio_rb32(pb); // dct_algo

            avio_rb32(pb); // strict_std_compliance

            avio_rb32(pb); // max_b_frames

            avio_rb32(pb); // mpeg_quant

            avio_rb32(pb); // intra_dc_precision

            avio_rb32(pb); // me_method

            avio_rb32(pb); // mb_decision

            avio_rb32(pb); // nsse_weight

            avio_rb32(pb); // frame_skip_cmp

            avio_rb64(pb); // rc_buffer_aggressivity

            codecpar->codec_tag = avio_rb32(pb);

            avio_r8(pb);   // thread_count

            avio_rb32(pb); // coder_type

            avio_rb32(pb); // me_cmp

            avio_rb32(pb); // me_subpel_quality

            avio_rb32(pb); // me_range

            avio_rb32(pb); // keyint_min

            avio_rb32(pb); // scenechange_threshold

            avio_rb32(pb); // b_frame_strategy

            avio_rb64(pb); // qcompress

            avio_rb64(pb); // qblur

            avio_rb32(pb); // max_qdiff

            avio_rb32(pb); // refs

            break;

        case MKBETAG('S', 'T', 'A', 'U'):

            if (f_stau++) {

                ret = AVERROR(EINVAL);

                goto fail;

            }

            codecpar->sample_rate = avio_rb32(pb);

            VALIDATE_PARAMETER(sample_rate, "sample rate",        codecpar->sample_rate < 0)

            codecpar->channels = avio_rl16(pb);

            VALIDATE_PARAMETER(channels,    "number of channels", codecpar->channels < 0)

            codecpar->frame_size = avio_rl16(pb);

            VALIDATE_PARAMETER(frame_size,  "frame size",         codecpar->frame_size < 0)

            break;

        case MKBETAG('C', 'P', 'R', 'V'):

            if (f_cprv++) {

                ret = AVERROR(EINVAL);

                goto fail;

            }

            enc = avcodec_find_encoder(codecpar->codec_id);

            if (enc && enc->priv_data_size && enc->priv_class) {

                buffer = av_malloc(size + 1);

                if (!buffer) {

                    ret = AVERROR(ENOMEM);

                    goto fail;

                }

                avio_get_str(pb, size, buffer, size + 1);

                if ((ret = ffm_append_recommended_configuration(st, &buffer)) < 0)

                    goto fail;

            }

            break;

        case MKBETAG('S', '2', 'V', 'I'):

            if (f_stvi++ || !size) {

                ret = AVERROR(EINVAL);

                goto fail;

            }

            buffer = av_malloc(size);

            if (!buffer) {

                ret = AVERROR(ENOMEM);

                goto fail;

            }

            avio_get_str(pb, INT_MAX, buffer, size);

            // The lack of AVOptions support in AVCodecParameters makes this back and forth copying needed

            avcodec_parameters_to_context(dummy_codec, codecpar);

            av_set_options_string(dummy_codec, buffer, "=", ",");

            avcodec_parameters_from_context(codecpar, dummy_codec);

            if ((ret = ffm_append_recommended_configuration(st, &buffer)) < 0)

                goto fail;

            break;

        case MKBETAG('S', '2', 'A', 'U'):

            if (f_stau++ || !size) {

                ret = AVERROR(EINVAL);

                goto fail;

            }

            buffer = av_malloc(size);

            if (!buffer) {

                ret = AVERROR(ENOMEM);

                goto fail;

            }

            avio_get_str(pb, INT_MAX, buffer, size);

            // The lack of AVOptions support in AVCodecParameters makes this back and forth copying needed

            avcodec_parameters_to_context(dummy_codec, codecpar);

            av_set_options_string(dummy_codec, buffer, "=", ",");

            avcodec_parameters_from_context(codecpar, dummy_codec);

            if ((ret = ffm_append_recommended_configuration(st, &buffer)) < 0)

                goto fail;

            break;

        }

        avio_seek(pb, next, SEEK_SET);

    }



    /* get until end of block reached */

    while ((avio_tell(pb) % ffm->packet_size) != 0 && !pb->eof_reached)

        avio_r8(pb);



    /* init packet demux */

    ffm->packet_ptr = ffm->packet;

    ffm->packet_end = ffm->packet;

    ffm->frame_offset = 0;

    ffm->dts = 0;

    ffm->read_state = READ_HEADER;

    ffm->first_packet = 1;

    avcodec_free_context(&dummy_codec);

    return 0;

 fail:

    avcodec_free_context(&dummy_codec);

    return ret;

}
