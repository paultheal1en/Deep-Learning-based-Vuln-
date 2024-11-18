static int segment_end(AVFormatContext *oc, int write_trailer)

{

    int ret = 0;



    av_write_frame(oc, NULL); /* Flush any buffered data (fragmented mp4) */

    if (write_trailer)

        av_write_trailer(oc);

    avio_close(oc->pb);



    return ret;

}
