ogm_header(AVFormatContext *s, int idx)

{

    struct ogg *ogg = s->priv_data;

    struct ogg_stream *os = ogg->streams + idx;

    AVStream *st = s->streams[idx];

    const uint8_t *p = os->buf + os->pstart;

    uint64_t time_unit;

    uint64_t spu;

    uint32_t size;



    if(!(*p & 1))

        return 0;



    if(*p == 1) {

        p++;



        if(*p == 'v'){

            int tag;

            st->codec->codec_type = AVMEDIA_TYPE_VIDEO;

            p += 8;

            tag = bytestream_get_le32(&p);

            st->codec->codec_id = ff_codec_get_id(ff_codec_bmp_tags, tag);

            st->codec->codec_tag = tag;

        } else if (*p == 't') {

            st->codec->codec_type = AVMEDIA_TYPE_SUBTITLE;

            st->codec->codec_id = CODEC_ID_TEXT;

            p += 12;

        } else {

            uint8_t acid[5];

            int cid;

            st->codec->codec_type = AVMEDIA_TYPE_AUDIO;

            p += 8;

            bytestream_get_buffer(&p, acid, 4);

            acid[4] = 0;

            cid = strtol(acid, NULL, 16);

            st->codec->codec_id = ff_codec_get_id(ff_codec_wav_tags, cid);

            // our parser completely breaks AAC in Ogg

            if (st->codec->codec_id != CODEC_ID_AAC)

                st->need_parsing = AVSTREAM_PARSE_FULL;

        }



        size        = bytestream_get_le32(&p);

        size        = FFMIN(size, os->psize);

        time_unit   = bytestream_get_le64(&p);

        spu         = bytestream_get_le64(&p);

        p += 4;                     /* default_len */

        p += 8;                     /* buffersize + bits_per_sample */



        if(st->codec->codec_type == AVMEDIA_TYPE_VIDEO){

            st->codec->width = bytestream_get_le32(&p);

            st->codec->height = bytestream_get_le32(&p);

            avpriv_set_pts_info(st, 64, time_unit, spu * 10000000);

        } else {

            st->codec->channels = bytestream_get_le16(&p);

            p += 2;                 /* block_align */

            st->codec->bit_rate = bytestream_get_le32(&p) * 8;

            st->codec->sample_rate = spu * 10000000 / time_unit;

            avpriv_set_pts_info(st, 64, 1, st->codec->sample_rate);

            if (size >= 56 && st->codec->codec_id == CODEC_ID_AAC) {

                p += 4;

                size -= 4;

            }

            if (size > 52) {

                av_assert0(FF_INPUT_BUFFER_PADDING_SIZE <= 52);

                size -= 52;

                st->codec->extradata_size = size;

                st->codec->extradata = av_malloc(size + FF_INPUT_BUFFER_PADDING_SIZE);

                bytestream_get_buffer(&p, st->codec->extradata, size);

            }

        }

    } else if (*p == 3) {

        if (os->psize > 8)

            ff_vorbis_comment(s, &st->metadata, p+7, os->psize-8);

    }



    return 1;

}
