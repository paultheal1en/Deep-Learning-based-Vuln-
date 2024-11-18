static int au_write_trailer(AVFormatContext *s)

{

    AVIOContext *pb = s->pb;

    int64_t file_size;



    if (s->pb->seekable) {

        /* update file size */

        file_size = avio_tell(pb);

        avio_seek(pb, 8, SEEK_SET);

        avio_wb32(pb, (uint32_t)(file_size - 24));

        avio_seek(pb, file_size, SEEK_SET);

        avio_flush(pb);

    }



    return 0;

}
