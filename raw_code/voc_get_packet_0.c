voc_get_packet(AVFormatContext *s, AVPacket *pkt, AVStream *st, int max_size)

{

    VocDecContext *voc = s->priv_data;

    AVCodecContext *dec = st->codec;

    ByteIOContext *pb = s->pb;

    VocType type;

    int size, tmp_codec;

    int sample_rate = 0;

    int channels = 1;



    while (!voc->remaining_size) {

        type = get_byte(pb);

        if (type == VOC_TYPE_EOF)

            return AVERROR(EIO);

        voc->remaining_size = get_le24(pb);

        if (!voc->remaining_size) {

            if (url_is_streamed(s->pb))

                return AVERROR(EIO);

            voc->remaining_size = url_fsize(pb) - url_ftell(pb);

        }

        max_size -= 4;



        switch (type) {

        case VOC_TYPE_VOICE_DATA:

            dec->sample_rate = 1000000 / (256 - get_byte(pb));

            if (sample_rate)

                dec->sample_rate = sample_rate;

            dec->channels = channels;

            tmp_codec = ff_codec_get_id(ff_voc_codec_tags, get_byte(pb));

            if (dec->codec_id == CODEC_ID_NONE)

                dec->codec_id = tmp_codec;

            else if (dec->codec_id != tmp_codec)

                av_log(s, AV_LOG_WARNING, "Ignoring mid-stream change in audio codec\n");

            dec->bits_per_coded_sample = av_get_bits_per_sample(dec->codec_id);

            voc->remaining_size -= 2;

            max_size -= 2;

            channels = 1;

            break;



        case VOC_TYPE_VOICE_DATA_CONT:

            break;



        case VOC_TYPE_EXTENDED:

            sample_rate = get_le16(pb);

            get_byte(pb);

            channels = get_byte(pb) + 1;

            sample_rate = 256000000 / (channels * (65536 - sample_rate));

            voc->remaining_size = 0;

            max_size -= 4;

            break;



        case VOC_TYPE_NEW_VOICE_DATA:

            dec->sample_rate = get_le32(pb);

            dec->bits_per_coded_sample = get_byte(pb);

            dec->channels = get_byte(pb);

            tmp_codec = ff_codec_get_id(ff_voc_codec_tags, get_le16(pb));

            if (dec->codec_id == CODEC_ID_NONE)

                dec->codec_id = tmp_codec;

            else if (dec->codec_id != tmp_codec)

                av_log(s, AV_LOG_WARNING, "Ignoring mid-stream change in audio codec\n");

            url_fskip(pb, 4);

            voc->remaining_size -= 12;

            max_size -= 12;

            break;



        default:

            url_fskip(pb, voc->remaining_size);

            max_size -= voc->remaining_size;

            voc->remaining_size = 0;

            break;

        }

        if (dec->codec_id == CODEC_ID_NONE) {

            av_log(s, AV_LOG_ERROR, "Invalid codec_id\n");

            if (s->audio_codec_id == CODEC_ID_NONE) return AVERROR(EINVAL);

        }

    }



    dec->bit_rate = dec->sample_rate * dec->bits_per_coded_sample;



    if (max_size <= 0)

        max_size = 2048;

    size = FFMIN(voc->remaining_size, max_size);

    voc->remaining_size -= size;

    return av_get_packet(pb, pkt, size);

}
