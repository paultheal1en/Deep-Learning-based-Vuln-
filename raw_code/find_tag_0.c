static int64_t find_tag(AVIOContext *pb, uint32_t tag1)

{

    unsigned int tag;

    int64_t size;



    for (;;) {

        if (url_feof(pb))

            return AVERROR_EOF;

        size = next_tag(pb, &tag);

        if (tag == tag1)

            break;

        wav_seek_tag(pb, size, SEEK_CUR);

    }

    return size;

}
