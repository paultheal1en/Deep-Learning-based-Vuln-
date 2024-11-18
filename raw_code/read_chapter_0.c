static void read_chapter(AVFormatContext *s, AVIOContext *pb, int len, char *ttag, ID3v2ExtraMeta **extra_meta)

{

    AVRational time_base = {1, 1000};

    uint32_t start, end;

    AVChapter *chapter;

    uint8_t *dst = NULL;

    int taglen;

    char tag[5];



    decode_str(s, pb, 0, &dst, &len);

    if (len < 16)

        return;



    start = avio_rb32(pb);

    end   = avio_rb32(pb);

    avio_skip(pb, 8);



    chapter = avpriv_new_chapter(s, s->nb_chapters + 1, time_base, start, end, dst);

    if (!chapter) {

        av_free(dst);

        return;

    }



    len -= 16;

    while (len > 10) {

        avio_read(pb, tag, 4);

        tag[4] = 0;

        taglen = avio_rb32(pb);

        avio_skip(pb, 2);

        len -= 10;

        if (taglen < 0 || taglen > len) {

            av_free(dst);

            return;

        }

        if (tag[0] == 'T')

            read_ttag(s, pb, taglen, &chapter->metadata, tag);

        else

            avio_skip(pb, taglen);

        len -= taglen;

    }



    ff_metadata_conv(&chapter->metadata, NULL, ff_id3v2_34_metadata_conv);

    ff_metadata_conv(&chapter->metadata, NULL, ff_id3v2_4_metadata_conv);

    av_free(dst);

}
