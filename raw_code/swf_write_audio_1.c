static int swf_write_audio(AVFormatContext *s, const uint8_t *buf, int size)

{

    ByteIOContext *pb = &s->pb;



    put_swf_tag(s, TAG_STREAMBLOCK | TAG_LONG);



    put_buffer(pb, buf, size);

    

    put_swf_end_tag(s);

    put_flush_packet(&s->pb);

    return 0;

}
