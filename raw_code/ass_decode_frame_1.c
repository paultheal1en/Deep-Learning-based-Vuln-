static int ass_decode_frame(AVCodecContext *avctx, void *data, int *got_sub_ptr,

                            AVPacket *avpkt)

{

    const char *ptr = avpkt->data;

    int len, size = avpkt->size;



    while (size > 0) {

        ASSDialog *dialog = ff_ass_split_dialog(avctx->priv_data, ptr, 0, NULL);

        int duration = dialog->end - dialog->start;

        len = ff_ass_add_rect(data, ptr, 0, duration, 1);

        if (len < 0)

            return len;

        ptr  += len;

        size -= len;

    }



    *got_sub_ptr = avpkt->size > 0;

    return avpkt->size;

}
