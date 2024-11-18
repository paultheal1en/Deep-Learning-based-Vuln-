static int queue_attached_pictures(AVFormatContext *s)

{

    int i;

    for (i = 0; i < s->nb_streams; i++)

        if (s->streams[i]->disposition & AV_DISPOSITION_ATTACHED_PIC &&

            s->streams[i]->discard < AVDISCARD_ALL) {

            AVPacket copy = s->streams[i]->attached_pic;

            copy.buf = av_buffer_ref(copy.buf);

            if (!copy.buf)

                return AVERROR(ENOMEM);



            add_to_pktbuf(&s->internal->raw_packet_buffer, &copy,

                          &s->internal->raw_packet_buffer_end);

        }

    return 0;

}
