static int aa_read_header(AVFormatContext *s)

{

    int i, j, idx, largest_idx = -1;

    uint32_t nkey, nval, toc_size, npairs, header_seed, start;

    char key[128], val[128], codec_name[64] = {0};

    uint8_t output[24], dst[8], src[8];

    int64_t largest_size = -1, current_size = -1;

    struct toc_entry {

        uint32_t offset;

        uint32_t size;

    } TOC[MAX_TOC_ENTRIES];

    uint32_t header_key_part[4];

    uint8_t header_key[16];

    AADemuxContext *c = s->priv_data;

    AVIOContext *pb = s->pb;

    AVStream *st;



    /* parse .aa header */

    avio_skip(pb, 4); // file size

    avio_skip(pb, 4); // magic string

    toc_size = avio_rb32(pb); // TOC size

    avio_skip(pb, 4); // unidentified integer

    if (toc_size > MAX_TOC_ENTRIES)

        return AVERROR_INVALIDDATA;

    for (i = 0; i < toc_size; i++) { // read TOC

        avio_skip(pb, 4); // TOC entry index

        TOC[i].offset = avio_rb32(pb); // block offset

        TOC[i].size = avio_rb32(pb); // block size

    }

    avio_skip(pb, 24); // header termination block (ignored)

    npairs = avio_rb32(pb); // read dictionary entries

    if (npairs > MAX_DICTIONARY_ENTRIES)

        return AVERROR_INVALIDDATA;

    for (i = 0; i < npairs; i++) {

        memset(val, 0, sizeof(val));

        memset(key, 0, sizeof(key));

        avio_skip(pb, 1); // unidentified integer

        nkey = avio_rb32(pb); // key string length

        nval = avio_rb32(pb); // value string length

        if (nkey > sizeof(key)) {

            avio_skip(pb, nkey);

        } else {

            avio_read(pb, key, nkey); // key string

        }

        if (nval > sizeof(val)) {

            avio_skip(pb, nval);

        } else {

            avio_read(pb, val, nval); // value string

        }

        if (!strcmp(key, "codec")) {

            av_log(s, AV_LOG_DEBUG, "Codec is <%s>\n", val);

            strncpy(codec_name, val, sizeof(codec_name) - 1);

        }

        if (!strcmp(key, "HeaderSeed")) {

            av_log(s, AV_LOG_DEBUG, "HeaderSeed is <%s>\n", val);

            header_seed = atoi(val);

        }

        if (!strcmp(key, "HeaderKey")) { // this looks like "1234567890 1234567890 1234567890 1234567890"

            av_log(s, AV_LOG_DEBUG, "HeaderKey is <%s>\n", val);

            sscanf(val, "%u%u%u%u", &header_key_part[0], &header_key_part[1], &header_key_part[2], &header_key_part[3]);

            for (idx = 0; idx < 4; idx++) {

                AV_WB32(&header_key[idx * 4], header_key_part[idx]); // convert each part to BE!

            }

            av_log(s, AV_LOG_DEBUG, "Processed HeaderKey is ");

            for (i = 0; i < 16; i++)

                av_log(s, AV_LOG_DEBUG, "%02x", header_key[i]);

            av_log(s, AV_LOG_DEBUG, "\n");

        }

    }



    /* verify fixed key */

    if (c->aa_fixed_key_len != 16) {

        av_log(s, AV_LOG_ERROR, "aa_fixed_key value needs to be 16 bytes!\n");

        return AVERROR(EINVAL);

    }



    /* verify codec */

    if ((c->codec_second_size = get_second_size(codec_name)) == -1) {

        av_log(s, AV_LOG_ERROR, "unknown codec <%s>!\n", codec_name);

        return AVERROR(EINVAL);

    }



    /* decryption key derivation */

    c->tea_ctx = av_tea_alloc();

    if (!c->tea_ctx)

        return AVERROR(ENOMEM);

    av_tea_init(c->tea_ctx, c->aa_fixed_key, 16);

    output[0] = output[1] = 0; // purely for padding purposes

    memcpy(output + 2, header_key, 16);

    idx = 0;

    for (i = 0; i < 3; i++) { // TEA CBC with weird mixed endianness

        AV_WB32(src, header_seed);

        AV_WB32(src + 4, header_seed + 1);

        header_seed += 2;

        av_tea_crypt(c->tea_ctx, dst, src, 1, NULL, 0); // TEA ECB encrypt

        for (j = 0; j < TEA_BLOCK_SIZE && idx < 18; j+=1, idx+=1) {

            output[idx] = output[idx] ^ dst[j];

        }

    }

    memcpy(c->file_key, output + 2, 16); // skip first 2 bytes of output

    av_log(s, AV_LOG_DEBUG, "File key is ");

    for (i = 0; i < 16; i++)

        av_log(s, AV_LOG_DEBUG, "%02x", c->file_key[i]);

    av_log(s, AV_LOG_DEBUG, "\n");



    /* decoder setup */

    st = avformat_new_stream(s, NULL);

    if (!st) {

        av_freep(&c->tea_ctx);

        return AVERROR(ENOMEM);

    }

    st->codec->codec_type = AVMEDIA_TYPE_AUDIO;

    if (!strcmp(codec_name, "mp332")) {

        st->codec->codec_id = AV_CODEC_ID_MP3;

        st->codec->sample_rate = 22050;

        st->need_parsing = AVSTREAM_PARSE_FULL_RAW;

        st->start_time = 0;

    } else if (!strcmp(codec_name, "acelp85")) {

        st->codec->codec_id = AV_CODEC_ID_SIPR;

        st->codec->block_align = 19;

        st->codec->channels = 1;

        st->codec->sample_rate = 8500;

    } else if (!strcmp(codec_name, "acelp16")) {

        st->codec->codec_id = AV_CODEC_ID_SIPR;

        st->codec->block_align = 20;

        st->codec->channels = 1;

        st->codec->sample_rate = 16000;

    }



    /* determine, and jump to audio start offset */

    for (i = 1; i < toc_size; i++) { // skip the first entry!

        current_size = TOC[i].size;

        if (current_size > largest_size) {

            largest_idx = i;

            largest_size = current_size;

        }

    }

    start = TOC[largest_idx].offset;

    avio_seek(pb, start, SEEK_SET);

    c->current_chapter_size = 0;



    return 0;

}
