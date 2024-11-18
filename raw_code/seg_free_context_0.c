static void seg_free_context(SegmentContext *seg)

{

    avio_closep(&seg->pb);

    avformat_free_context(seg->avf);

    seg->avf = NULL;

}
