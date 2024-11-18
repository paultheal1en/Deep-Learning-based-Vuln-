int ff_get_wav_header(AVIOContext *pb, AVCodecContext *codec, int size)

{

    int id;



    id = avio_rl16(pb);

    codec->codec_type = AVMEDIA_TYPE_AUDIO;

    codec->channels = avio_rl16(pb);

    codec->sample_rate = avio_rl32(pb);

    codec->bit_rate = avio_rl32(pb) * 8;

    codec->block_align = avio_rl16(pb);

    if (size == 14) {  /* We're dealing with plain vanilla WAVEFORMAT */

        codec->bits_per_coded_sample = 8;

    }else

        codec->bits_per_coded_sample = avio_rl16(pb);

    if (id == 0xFFFE) {

        codec->codec_tag = 0;

    } else {

        codec->codec_tag = id;

        codec->codec_id = ff_wav_codec_get_id(id, codec->bits_per_coded_sample);

    }

    if (size >= 18) {  /* We're obviously dealing with WAVEFORMATEX */

        int cbSize = avio_rl16(pb); /* cbSize */

        size -= 18;

        cbSize = FFMIN(size, cbSize);

        if (cbSize >= 22 && id == 0xfffe) { /* WAVEFORMATEXTENSIBLE */

            ff_asf_guid subformat;

            codec->bits_per_coded_sample = avio_rl16(pb);

            codec->channel_layout = avio_rl32(pb); /* dwChannelMask */

            ff_get_guid(pb, &subformat);

            if (!memcmp(subformat + 4, (const uint8_t[]){FF_MEDIASUBTYPE_BASE_GUID}, 12)) {

                codec->codec_tag = AV_RL32(subformat);

                codec->codec_id = ff_wav_codec_get_id(codec->codec_tag, codec->bits_per_coded_sample);

            } else {

                codec->codec_id = ff_codec_guid_get_id(ff_codec_wav_guids, subformat);

                if (!codec->codec_id)

                    av_log(codec, AV_LOG_WARNING, "unknown subformat:"FF_PRI_GUID"\n", FF_ARG_GUID(subformat));

            }

            cbSize -= 22;

            size -= 22;

        }

        codec->extradata_size = cbSize;

        if (cbSize > 0) {

            av_free(codec->extradata);

            codec->extradata = av_mallocz(codec->extradata_size + FF_INPUT_BUFFER_PADDING_SIZE);

            if (!codec->extradata)

                return AVERROR(ENOMEM);

            avio_read(pb, codec->extradata, codec->extradata_size);

            size -= cbSize;

        }



        /* It is possible for the chunk to contain garbage at the end */

        if (size > 0)

            avio_skip(pb, size);

    }

    if (codec->codec_id == CODEC_ID_AAC_LATM) {

        /* channels and sample_rate values are those prior to applying SBR and/or PS */

        codec->channels    = 0;

        codec->sample_rate = 0;

    }

    /* override bits_per_coded_sample for G.726 */

    if (codec->codec_id == CODEC_ID_ADPCM_G726)

        codec->bits_per_coded_sample = codec->bit_rate / codec->sample_rate;



    return 0;

}
