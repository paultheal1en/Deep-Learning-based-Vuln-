int ff_read_riff_info(AVFormatContext *s, int64_t size)

{

    int64_t start, end, cur;

    AVIOContext *pb = s->pb;



    start = avio_tell(pb);

    end = start + size;



    while ((cur = avio_tell(pb)) >= 0 && cur <= end - 8 /* = tag + size */) {

        uint32_t chunk_code;

        int64_t chunk_size;

        char key[5] = {0};

        char *value;



        chunk_code = avio_rl32(pb);

        chunk_size = avio_rl32(pb);

        if (chunk_size > end || end - chunk_size < cur || chunk_size == UINT_MAX) {

            av_log(s, AV_LOG_ERROR, "too big INFO subchunk\n");

            return AVERROR_INVALIDDATA;

        }



        chunk_size += (chunk_size & 1);



        value = av_malloc(chunk_size + 1);

        if (!value) {

            av_log(s, AV_LOG_ERROR, "out of memory, unable to read INFO tag\n");

            return AVERROR(ENOMEM);

        }



        AV_WL32(key, chunk_code);



        if (avio_read(pb, value, chunk_size) != chunk_size) {

            av_freep(key);

            av_freep(value);

            av_log(s, AV_LOG_ERROR, "premature end of file while reading INFO tag\n");

            return AVERROR_INVALIDDATA;

        }



        value[chunk_size] = 0;



        av_dict_set(&s->metadata, key, value, AV_DICT_DONT_STRDUP_VAL);

    }



    return 0;

}
