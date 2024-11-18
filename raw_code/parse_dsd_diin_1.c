static int parse_dsd_diin(AVFormatContext *s, AVStream *st, uint64_t eof)

{

    AVIOContext *pb = s->pb;



    while (avio_tell(pb) + 12 <= eof) {

        uint32_t tag      = avio_rl32(pb);

        uint64_t size     = avio_rb64(pb);

        uint64_t orig_pos = avio_tell(pb);

        const char * metadata_tag = NULL;



        switch(tag) {

        case MKTAG('D','I','A','R'): metadata_tag = "artist"; break;

        case MKTAG('D','I','T','I'): metadata_tag = "title";  break;

        }



        if (metadata_tag && size > 4) {

            unsigned int tag_size = avio_rb32(pb);

            int ret = get_metadata(s, metadata_tag, FFMIN(tag_size, size - 4));

            if (ret < 0) {

                av_log(s, AV_LOG_ERROR, "cannot allocate metadata tag %s!\n", metadata_tag);

                return ret;

            }

        }



        avio_skip(pb, size - (avio_tell(pb) - orig_pos) + (size & 1));

    }



    return 0;

}
