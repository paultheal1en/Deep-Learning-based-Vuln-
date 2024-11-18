static int asf_read_header(AVFormatContext *s, AVFormatParameters *ap)

{

    ASFContext *asf = s->priv_data;

    ff_asf_guid g;

    ByteIOContext *pb = s->pb;

    AVStream *st;

    ASFStream *asf_st;

    int size, i;

    int64_t gsize;

    AVRational dar[128];

    uint32_t bitrate[128];



    memset(dar, 0, sizeof(dar));

    memset(bitrate, 0, sizeof(bitrate));



    get_guid(pb, &g);

    if (guidcmp(&g, &ff_asf_header))

        return -1;

    get_le64(pb);

    get_le32(pb);

    get_byte(pb);

    get_byte(pb);

    memset(&asf->asfid2avid, -1, sizeof(asf->asfid2avid));

    for(;;) {

        uint64_t gpos= url_ftell(pb);

        get_guid(pb, &g);

        gsize = get_le64(pb);

        dprintf(s, "%08"PRIx64": ", gpos);

        print_guid(&g);

        dprintf(s, "  size=0x%"PRIx64"\n", gsize);

        if (!guidcmp(&g, &ff_asf_data_header)) {

            asf->data_object_offset = url_ftell(pb);

            // if not streaming, gsize is not unlimited (how?), and there is enough space in the file..

            if (!(asf->hdr.flags & 0x01) && gsize >= 100) {

                asf->data_object_size = gsize - 24;

            } else {

                asf->data_object_size = (uint64_t)-1;

            }

            break;

        }

        if (gsize < 24)

            return -1;

        if (!guidcmp(&g, &ff_asf_file_header)) {

            get_guid(pb, &asf->hdr.guid);

            asf->hdr.file_size          = get_le64(pb);

            asf->hdr.create_time        = get_le64(pb);

            asf->nb_packets             = get_le64(pb);

            asf->hdr.play_time          = get_le64(pb);

            asf->hdr.send_time          = get_le64(pb);

            asf->hdr.preroll            = get_le32(pb);

            asf->hdr.ignore             = get_le32(pb);

            asf->hdr.flags              = get_le32(pb);

            asf->hdr.min_pktsize        = get_le32(pb);

            asf->hdr.max_pktsize        = get_le32(pb);

            asf->hdr.max_bitrate        = get_le32(pb);

            s->packet_size = asf->hdr.max_pktsize;

        } else if (!guidcmp(&g, &ff_asf_stream_header)) {

            enum AVMediaType type;

            int type_specific_size, sizeX;

            uint64_t total_size;

            unsigned int tag1;

            int64_t pos1, pos2, start_time;

            int test_for_ext_stream_audio, is_dvr_ms_audio=0;



            if (s->nb_streams == ASF_MAX_STREAMS) {

                av_log(s, AV_LOG_ERROR, "too many streams\n");

                return AVERROR(EINVAL);

            }



            pos1 = url_ftell(pb);



            st = av_new_stream(s, 0);

            if (!st)

                return AVERROR(ENOMEM);

            av_set_pts_info(st, 32, 1, 1000); /* 32 bit pts in ms */

            asf_st = av_mallocz(sizeof(ASFStream));

            if (!asf_st)

                return AVERROR(ENOMEM);

            st->priv_data = asf_st;

            start_time = asf->hdr.preroll;



            asf_st->stream_language_index = 128; // invalid stream index means no language info



            if(!(asf->hdr.flags & 0x01)) { // if we aren't streaming...

                st->duration = asf->hdr.play_time /

                    (10000000 / 1000) - start_time;

            }

            get_guid(pb, &g);



            test_for_ext_stream_audio = 0;

            if (!guidcmp(&g, &ff_asf_audio_stream)) {

                type = AVMEDIA_TYPE_AUDIO;

            } else if (!guidcmp(&g, &ff_asf_video_stream)) {

                type = AVMEDIA_TYPE_VIDEO;

            } else if (!guidcmp(&g, &ff_asf_command_stream)) {

                type = AVMEDIA_TYPE_DATA;

            } else if (!guidcmp(&g, &ff_asf_ext_stream_embed_stream_header)) {

                test_for_ext_stream_audio = 1;

                type = AVMEDIA_TYPE_UNKNOWN;

            } else {

                return -1;

            }

            get_guid(pb, &g);

            total_size = get_le64(pb);

            type_specific_size = get_le32(pb);

            get_le32(pb);

            st->id = get_le16(pb) & 0x7f; /* stream id */

            // mapping of asf ID to AV stream ID;

            asf->asfid2avid[st->id] = s->nb_streams - 1;



            get_le32(pb);



            if (test_for_ext_stream_audio) {

                get_guid(pb, &g);

                if (!guidcmp(&g, &ff_asf_ext_stream_audio_stream)) {

                    type = AVMEDIA_TYPE_AUDIO;

                    is_dvr_ms_audio=1;

                    get_guid(pb, &g);

                    get_le32(pb);

                    get_le32(pb);

                    get_le32(pb);

                    get_guid(pb, &g);

                    get_le32(pb);

                }

            }



            st->codec->codec_type = type;

            if (type == AVMEDIA_TYPE_AUDIO) {

                ff_get_wav_header(pb, st->codec, type_specific_size);

                if (is_dvr_ms_audio) {

                    // codec_id and codec_tag are unreliable in dvr_ms

                    // files. Set them later by probing stream.

                    st->codec->codec_id = CODEC_ID_PROBE;

                    st->codec->codec_tag = 0;

                }

                if (st->codec->codec_id == CODEC_ID_AAC) {

                    st->need_parsing = AVSTREAM_PARSE_NONE;

                } else {

                    st->need_parsing = AVSTREAM_PARSE_FULL;

                }

                /* We have to init the frame size at some point .... */

                pos2 = url_ftell(pb);

                if (gsize >= (pos2 + 8 - pos1 + 24)) {

                    asf_st->ds_span = get_byte(pb);

                    asf_st->ds_packet_size = get_le16(pb);

                    asf_st->ds_chunk_size = get_le16(pb);

                    get_le16(pb); //ds_data_size

                    get_byte(pb); //ds_silence_data

                }

                //printf("Descrambling: ps:%d cs:%d ds:%d s:%d  sd:%d\n",

                //       asf_st->ds_packet_size, asf_st->ds_chunk_size,

                //       asf_st->ds_data_size, asf_st->ds_span, asf_st->ds_silence_data);

                if (asf_st->ds_span > 1) {

                    if (!asf_st->ds_chunk_size

                        || (asf_st->ds_packet_size/asf_st->ds_chunk_size <= 1)

                        || asf_st->ds_packet_size % asf_st->ds_chunk_size)

                        asf_st->ds_span = 0; // disable descrambling

                }

                switch (st->codec->codec_id) {

                case CODEC_ID_MP3:

                    st->codec->frame_size = MPA_FRAME_SIZE;

                    break;

                case CODEC_ID_PCM_S16LE:

                case CODEC_ID_PCM_S16BE:

                case CODEC_ID_PCM_U16LE:

                case CODEC_ID_PCM_U16BE:

                case CODEC_ID_PCM_S8:

                case CODEC_ID_PCM_U8:

                case CODEC_ID_PCM_ALAW:

                case CODEC_ID_PCM_MULAW:

                    st->codec->frame_size = 1;

                    break;

                default:

                    /* This is probably wrong, but it prevents a crash later */

                    st->codec->frame_size = 1;

                    break;

                }

            } else if (type == AVMEDIA_TYPE_VIDEO) {

                get_le32(pb);

                get_le32(pb);

                get_byte(pb);

                size = get_le16(pb); /* size */

                sizeX= get_le32(pb); /* size */

                st->codec->width = get_le32(pb);

                st->codec->height = get_le32(pb);

                /* not available for asf */

                get_le16(pb); /* panes */

                st->codec->bits_per_coded_sample = get_le16(pb); /* depth */

                tag1 = get_le32(pb);

                url_fskip(pb, 20);

//                av_log(s, AV_LOG_DEBUG, "size:%d tsize:%d sizeX:%d\n", size, total_size, sizeX);

                size= sizeX;

                if (size > 40) {

                    st->codec->extradata_size = size - 40;

                    st->codec->extradata = av_mallocz(st->codec->extradata_size + FF_INPUT_BUFFER_PADDING_SIZE);

                    get_buffer(pb, st->codec->extradata, st->codec->extradata_size);

                }



                /* Extract palette from extradata if bpp <= 8 */

                /* This code assumes that extradata contains only palette */

                /* This is true for all paletted codecs implemented in ffmpeg */

                if (st->codec->extradata_size && (st->codec->bits_per_coded_sample <= 8)) {

                    st->codec->palctrl = av_mallocz(sizeof(AVPaletteControl));

#if HAVE_BIGENDIAN

                    for (i = 0; i < FFMIN(st->codec->extradata_size, AVPALETTE_SIZE)/4; i++)

                        st->codec->palctrl->palette[i] = bswap_32(((uint32_t*)st->codec->extradata)[i]);

#else

                    memcpy(st->codec->palctrl->palette, st->codec->extradata,

                           FFMIN(st->codec->extradata_size, AVPALETTE_SIZE));

#endif

                    st->codec->palctrl->palette_changed = 1;

                }



                st->codec->codec_tag = tag1;

                st->codec->codec_id = ff_codec_get_id(ff_codec_bmp_tags, tag1);

                if(tag1 == MKTAG('D', 'V', 'R', ' '))

                    st->need_parsing = AVSTREAM_PARSE_FULL;

                if(st->codec->codec_id == CODEC_ID_H264)

                    st->need_parsing = AVSTREAM_PARSE_FULL_ONCE;

            }

            pos2 = url_ftell(pb);

            url_fskip(pb, gsize - (pos2 - pos1 + 24));

        } else if (!guidcmp(&g, &ff_asf_comment_header)) {

            int len1, len2, len3, len4, len5;



            len1 = get_le16(pb);

            len2 = get_le16(pb);

            len3 = get_le16(pb);

            len4 = get_le16(pb);

            len5 = get_le16(pb);

            get_tag(s, "title"    , 0, len1);

            get_tag(s, "author"   , 0, len2);

            get_tag(s, "copyright", 0, len3);

            get_tag(s, "comment"  , 0, len4);

            url_fskip(pb, len5);

        } else if (!guidcmp(&g, &stream_bitrate_guid)) {

            int stream_count = get_le16(pb);

            int j;



//            av_log(s, AV_LOG_ERROR, "stream bitrate properties\n");

//            av_log(s, AV_LOG_ERROR, "streams %d\n", streams);

            for(j = 0; j < stream_count; j++) {

                int flags, bitrate, stream_id;



                flags= get_le16(pb);

                bitrate= get_le32(pb);

                stream_id= (flags & 0x7f);

//                av_log(s, AV_LOG_ERROR, "flags: 0x%x stream id %d, bitrate %d\n", flags, stream_id, bitrate);

                asf->stream_bitrates[stream_id]= bitrate;

            }

        } else if (!guidcmp(&g, &ff_asf_language_guid)) {

            int j;

            int stream_count = get_le16(pb);

            for(j = 0; j < stream_count; j++) {

                char lang[6];

                unsigned int lang_len = get_byte(pb);

                get_str16_nolen(pb, lang_len, lang, sizeof(lang));

                if (j < 128)

                    av_strlcpy(asf->stream_languages[j], lang, sizeof(*asf->stream_languages));

            }

        } else if (!guidcmp(&g, &ff_asf_extended_content_header)) {

            int desc_count, i;



            desc_count = get_le16(pb);

            for(i=0;i<desc_count;i++) {

                    int name_len,value_type,value_len;

                    char name[1024];



                    name_len = get_le16(pb);

                    if (name_len%2)     // must be even, broken lavf versions wrote len-1

                        name_len += 1;

                    get_str16_nolen(pb, name_len, name, sizeof(name));

                    value_type = get_le16(pb);

                    value_len  = get_le16(pb);

                    if (!value_type && value_len%2)

                        value_len += 1;

                    get_tag(s, name, value_type, value_len);

            }

        } else if (!guidcmp(&g, &ff_asf_metadata_header)) {

            int n, stream_num, name_len, value_len, value_type, value_num;

            n = get_le16(pb);



            for(i=0;i<n;i++) {

                char name[1024];



                get_le16(pb); //lang_list_index

                stream_num= get_le16(pb);

                name_len=   get_le16(pb);

                value_type= get_le16(pb);

                value_len=  get_le32(pb);



                get_str16_nolen(pb, name_len, name, sizeof(name));

//av_log(s, AV_LOG_ERROR, "%d %d %d %d %d <%s>\n", i, stream_num, name_len, value_type, value_len, name);

                value_num= get_le16(pb);//we should use get_value() here but it does not work 2 is le16 here but le32 elsewhere

                url_fskip(pb, value_len - 2);



                if(stream_num<128){

                    if     (!strcmp(name, "AspectRatioX")) dar[stream_num].num= value_num;

                    else if(!strcmp(name, "AspectRatioY")) dar[stream_num].den= value_num;

                }

            }

        } else if (!guidcmp(&g, &ff_asf_ext_stream_header)) {

            int ext_len, payload_ext_ct, stream_ct;

            uint32_t ext_d, leak_rate, stream_num;

            unsigned int stream_languageid_index;



            get_le64(pb); // starttime

            get_le64(pb); // endtime

            leak_rate = get_le32(pb); // leak-datarate

            get_le32(pb); // bucket-datasize

            get_le32(pb); // init-bucket-fullness

            get_le32(pb); // alt-leak-datarate

            get_le32(pb); // alt-bucket-datasize

            get_le32(pb); // alt-init-bucket-fullness

            get_le32(pb); // max-object-size

            get_le32(pb); // flags (reliable,seekable,no_cleanpoints?,resend-live-cleanpoints, rest of bits reserved)

            stream_num = get_le16(pb); // stream-num



            stream_languageid_index = get_le16(pb); // stream-language-id-index

            if (stream_num < 128)

                asf->streams[stream_num].stream_language_index = stream_languageid_index;



            get_le64(pb); // avg frametime in 100ns units

            stream_ct = get_le16(pb); //stream-name-count

            payload_ext_ct = get_le16(pb); //payload-extension-system-count



            if (stream_num < 128)

                bitrate[stream_num] = leak_rate;



            for (i=0; i<stream_ct; i++){

                get_le16(pb);

                ext_len = get_le16(pb);

                url_fseek(pb, ext_len, SEEK_CUR);

            }



            for (i=0; i<payload_ext_ct; i++){

                get_guid(pb, &g);

                ext_d=get_le16(pb);

                ext_len=get_le32(pb);

                url_fseek(pb, ext_len, SEEK_CUR);

            }



            // there could be a optional stream properties object to follow

            // if so the next iteration will pick it up

            continue;

        } else if (!guidcmp(&g, &ff_asf_head1_guid)) {

            int v1, v2;

            get_guid(pb, &g);

            v1 = get_le32(pb);

            v2 = get_le16(pb);

            continue;

        } else if (!guidcmp(&g, &ff_asf_marker_header)) {

            int i, count, name_len;

            char name[1024];



            get_le64(pb);            // reserved 16 bytes

            get_le64(pb);            // ...

            count = get_le32(pb);    // markers count

            get_le16(pb);            // reserved 2 bytes

            name_len = get_le16(pb); // name length

            for(i=0;i<name_len;i++){

                get_byte(pb); // skip the name

            }



            for(i=0;i<count;i++){

                int64_t pres_time;

                int name_len;



                get_le64(pb);             // offset, 8 bytes

                pres_time = get_le64(pb); // presentation time

                get_le16(pb);             // entry length

                get_le32(pb);             // send time

                get_le32(pb);             // flags

                name_len = get_le32(pb);  // name length

                get_str16_nolen(pb, name_len * 2, name, sizeof(name));

                ff_new_chapter(s, i, (AVRational){1, 10000000}, pres_time, AV_NOPTS_VALUE, name );

            }

#if 0

        } else if (!guidcmp(&g, &ff_asf_codec_comment_header)) {

            int len, v1, n, num;

            char str[256], *q;

            char tag[16];



            get_guid(pb, &g);

            print_guid(&g);



            n = get_le32(pb);

            for(i=0;i<n;i++) {

                num = get_le16(pb); /* stream number */

                get_str16(pb, str, sizeof(str));

                get_str16(pb, str, sizeof(str));

                len = get_le16(pb);

                q = tag;

                while (len > 0) {

                    v1 = get_byte(pb);

                    if ((q - tag) < sizeof(tag) - 1)

                        *q++ = v1;

                    len--;

                }

                *q = '\0';

            }

#endif

        } else if (url_feof(pb)) {

            return -1;

        } else {

            if (!s->keylen) {

                if (!guidcmp(&g, &ff_asf_content_encryption)) {

                    av_log(s, AV_LOG_WARNING, "DRM protected stream detected, decoding will likely fail!\n");

                } else if (!guidcmp(&g, &ff_asf_ext_content_encryption)) {

                    av_log(s, AV_LOG_WARNING, "Ext DRM protected stream detected, decoding will likely fail!\n");

                } else if (!guidcmp(&g, &ff_asf_digital_signature)) {

                    av_log(s, AV_LOG_WARNING, "Digital signature detected, decoding will likely fail!\n");

                }

            }

        }

        if(url_ftell(pb) != gpos + gsize)

            av_log(s, AV_LOG_DEBUG, "gpos mismatch our pos=%"PRIu64", end=%"PRIu64"\n", url_ftell(pb)-gpos, gsize);

        url_fseek(pb, gpos + gsize, SEEK_SET);

    }

    get_guid(pb, &g);

    get_le64(pb);

    get_byte(pb);

    get_byte(pb);

    if (url_feof(pb))

        return -1;

    asf->data_offset = url_ftell(pb);

    asf->packet_size_left = 0;





    for(i=0; i<128; i++){

        int stream_num= asf->asfid2avid[i];

        if(stream_num>=0){

            AVStream *st = s->streams[stream_num];

            if (!st->codec->bit_rate)

                st->codec->bit_rate = bitrate[i];

            if (dar[i].num > 0 && dar[i].den > 0)

                av_reduce(&st->sample_aspect_ratio.num,

                          &st->sample_aspect_ratio.den,

                          dar[i].num, dar[i].den, INT_MAX);

//av_log(s, AV_LOG_ERROR, "dar %d:%d sar=%d:%d\n", dar[i].num, dar[i].den, st->sample_aspect_ratio.num, st->sample_aspect_ratio.den);



            // copy and convert language codes to the frontend

            if (asf->streams[i].stream_language_index < 128) {

                const char *rfc1766 = asf->stream_languages[asf->streams[i].stream_language_index];

                if (rfc1766 && strlen(rfc1766) > 1) {

                    const char primary_tag[3] = { rfc1766[0], rfc1766[1], '\0' }; // ignore country code if any

                    const char *iso6392 = av_convert_lang_to(primary_tag, AV_LANG_ISO639_2_BIBL);

                    if (iso6392)

                        av_metadata_set2(&st->metadata, "language", iso6392, 0);

                }

            }

        }

    }



    return 0;

}
