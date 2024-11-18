static int asf_read_header(AVFormatContext *s)

{

    ASFContext *asf = s->priv_data;

    ff_asf_guid g;

    AVIOContext *pb = s->pb;

    int i;

    int64_t gsize;



    ff_get_guid(pb, &g);

    if (ff_guidcmp(&g, &ff_asf_header))

        return AVERROR_INVALIDDATA;

    avio_rl64(pb);

    avio_rl32(pb);

    avio_r8(pb);

    avio_r8(pb);

    memset(&asf->asfid2avid, -1, sizeof(asf->asfid2avid));



    for (i = 0; i<128; i++)

        asf->streams[i].stream_language_index = 128; // invalid stream index means no language info



    for (;;) {

        uint64_t gpos = avio_tell(pb);

        ff_get_guid(pb, &g);

        gsize = avio_rl64(pb);

        print_guid(&g);

        if (!ff_guidcmp(&g, &ff_asf_data_header)) {

            asf->data_object_offset = avio_tell(pb);

            /* If not streaming, gsize is not unlimited (how?),

             * and there is enough space in the file.. */

            if (!(asf->hdr.flags & 0x01) && gsize >= 100)

                asf->data_object_size = gsize - 24;

            else

                asf->data_object_size = (uint64_t)-1;

            break;

        }

        if (gsize < 24)

            return AVERROR_INVALIDDATA;

        if (!ff_guidcmp(&g, &ff_asf_file_header)) {

            int ret = asf_read_file_properties(s, gsize);

            if (ret < 0)

                return ret;

        } else if (!ff_guidcmp(&g, &ff_asf_stream_header)) {

            int ret = asf_read_stream_properties(s, gsize);

            if (ret < 0)

                return ret;

        } else if (!ff_guidcmp(&g, &ff_asf_comment_header)) {

            asf_read_content_desc(s, gsize);

        } else if (!ff_guidcmp(&g, &ff_asf_language_guid)) {

            asf_read_language_list(s, gsize);

        } else if (!ff_guidcmp(&g, &ff_asf_extended_content_header)) {

            asf_read_ext_content_desc(s, gsize);

        } else if (!ff_guidcmp(&g, &ff_asf_metadata_header)) {

            asf_read_metadata(s, gsize);

        } else if (!ff_guidcmp(&g, &ff_asf_metadata_library_header)) {

            asf_read_metadata(s, gsize);

        } else if (!ff_guidcmp(&g, &ff_asf_ext_stream_header)) {

            asf_read_ext_stream_properties(s, gsize);



            // there could be a optional stream properties object to follow

            // if so the next iteration will pick it up

            continue;

        } else if (!ff_guidcmp(&g, &ff_asf_head1_guid)) {

            ff_get_guid(pb, &g);

            avio_skip(pb, 6);

            continue;

        } else if (!ff_guidcmp(&g, &ff_asf_marker_header)) {

            asf_read_marker(s, gsize);

        } else if (avio_feof(pb)) {

            return AVERROR_EOF;

        } else {

            if (!s->keylen) {

                if (!ff_guidcmp(&g, &ff_asf_content_encryption)) {

                    unsigned int len;

                    int ret;

                    AVPacket pkt;

                    av_log(s, AV_LOG_WARNING,

                           "DRM protected stream detected, decoding will likely fail!\n");

                    len= avio_rl32(pb);

                    av_log(s, AV_LOG_DEBUG, "Secret data:\n");



                    if ((ret = av_get_packet(pb, &pkt, len)) < 0)

                        return ret;

                    av_hex_dump_log(s, AV_LOG_DEBUG, pkt.data, pkt.size);

                    av_free_packet(&pkt);

                    len= avio_rl32(pb);

                    get_tag(s, "ASF_Protection_Type", -1, len, 32);

                    len= avio_rl32(pb);

                    get_tag(s, "ASF_Key_ID", -1, len, 32);

                    len= avio_rl32(pb);

                    get_tag(s, "ASF_License_URL", -1, len, 32);

                } else if (!ff_guidcmp(&g, &ff_asf_ext_content_encryption)) {

                    av_log(s, AV_LOG_WARNING,

                           "Ext DRM protected stream detected, decoding will likely fail!\n");

                    av_dict_set(&s->metadata, "encryption", "ASF Extended Content Encryption", 0);

                } else if (!ff_guidcmp(&g, &ff_asf_digital_signature)) {

                    av_log(s, AV_LOG_INFO, "Digital signature detected!\n");

                }

            }

        }

        if (avio_tell(pb) != gpos + gsize)

            av_log(s, AV_LOG_DEBUG,

                   "gpos mismatch our pos=%"PRIu64", end=%"PRId64"\n",

                   avio_tell(pb) - gpos, gsize);

        avio_seek(pb, gpos + gsize, SEEK_SET);

    }

    ff_get_guid(pb, &g);

    avio_rl64(pb);

    avio_r8(pb);

    avio_r8(pb);

    if (avio_feof(pb))

        return AVERROR_EOF;

    asf->data_offset      = avio_tell(pb);

    asf->packet_size_left = 0;



    for (i = 0; i < 128; i++) {

        int stream_num = asf->asfid2avid[i];

        if (stream_num >= 0) {

            AVStream *st = s->streams[stream_num];

            if (!st->codec->bit_rate)

                st->codec->bit_rate = asf->stream_bitrates[i];

            if (asf->dar[i].num > 0 && asf->dar[i].den > 0) {

                av_reduce(&st->sample_aspect_ratio.num,

                          &st->sample_aspect_ratio.den,

                          asf->dar[i].num, asf->dar[i].den, INT_MAX);

            } else if ((asf->dar[0].num > 0) && (asf->dar[0].den > 0) &&

                       // Use ASF container value if the stream doesn't set AR.

                       (st->codec->codec_type == AVMEDIA_TYPE_VIDEO))

                av_reduce(&st->sample_aspect_ratio.num,

                          &st->sample_aspect_ratio.den,

                          asf->dar[0].num, asf->dar[0].den, INT_MAX);



            av_log(s, AV_LOG_TRACE, "i=%d, st->codec->codec_type:%d, asf->dar %d:%d sar=%d:%d\n",

                    i, st->codec->codec_type, asf->dar[i].num, asf->dar[i].den,

                    st->sample_aspect_ratio.num, st->sample_aspect_ratio.den);



            // copy and convert language codes to the frontend

            if (asf->streams[i].stream_language_index < 128) {

                const char *rfc1766 = asf->stream_languages[asf->streams[i].stream_language_index];

                if (rfc1766 && strlen(rfc1766) > 1) {

                    const char primary_tag[3] = { rfc1766[0], rfc1766[1], '\0' }; // ignore country code if any

                    const char *iso6392       = av_convert_lang_to(primary_tag,

                                                                   AV_LANG_ISO639_2_BIBL);

                    if (iso6392)

                        av_dict_set(&st->metadata, "language", iso6392, 0);

                }

            }

        }

    }



    ff_metadata_conv(&s->metadata, NULL, ff_asf_metadata_conv);



    return 0;

}
