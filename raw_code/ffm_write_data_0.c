static void ffm_write_data(AVFormatContext *s,

                           const uint8_t *buf, int size,

                           int64_t pts, int header)

{

    FFMContext *ffm = s->priv_data;

    int len;



    if (header && ffm->frame_offset == 0) {

        ffm->frame_offset = ffm->packet_ptr - ffm->packet + FFM_HEADER_SIZE;

        ffm->pts = pts;

    }



    /* write as many packets as needed */

    while (size > 0) {

        len = ffm->packet_end - ffm->packet_ptr;

        if (len > size)

            len = size;

        memcpy(ffm->packet_ptr, buf, len);



        ffm->packet_ptr += len;

        buf += len;

        size -= len;

        if (ffm->packet_ptr >= ffm->packet_end) {

            /* special case : no pts in packet : we leave the current one */

            if (ffm->pts == 0)

                ffm->pts = pts;



            flush_packet(s);

        }

    }

}
