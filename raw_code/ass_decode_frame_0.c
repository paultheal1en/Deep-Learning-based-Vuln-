static int ass_decode_frame(AVCodecContext *avctx, void *data, int *got_sub_ptr,

                            AVPacket *avpkt)

{

    const char *ptr = avpkt->data;

    int len, size = avpkt->size;



    ff_ass_init(data);



    while (size > 0) {

        len = ff_ass_add_rect(data, ptr, 0, 0/* FIXME: duration */, 1);

        if (len < 0)

            return len;

        ptr  += len;

        size -= len;

    }



    *got_sub_ptr = avpkt->size > 0;

    return avpkt->size;

}
