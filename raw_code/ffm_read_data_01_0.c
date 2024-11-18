static int ffm_read_data(AVFormatContext *s,

                         uint8_t *buf, int size, int first)

{

    FFMContext *ffm = s->priv_data;

    ByteIOContext *pb = s->pb;

    int len, fill_size, size1, frame_offset;



    size1 = size;

    while (size > 0) {

    redo:

        len = ffm->packet_end - ffm->packet_ptr;

        if (len < 0)

            return -1;

        if (len > size)

            len = size;

        if (len == 0) {

            if (url_ftell(pb) == ffm->file_size)

                url_fseek(pb, ffm->packet_size, SEEK_SET);

    retry_read:

            get_be16(pb); /* PACKET_ID */

            fill_size = get_be16(pb);

            ffm->pts = get_be64(pb);

            ffm->first_frame_in_packet = 1;

            frame_offset = get_be16(pb);

            get_buffer(pb, ffm->packet, ffm->packet_size - FFM_HEADER_SIZE);

            ffm->packet_end = ffm->packet + (ffm->packet_size - FFM_HEADER_SIZE - fill_size);

            if (ffm->packet_end < ffm->packet)

                return -1;

            /* if first packet or resynchronization packet, we must

               handle it specifically */

            if (ffm->first_packet || (frame_offset & 0x8000)) {

                if (!frame_offset) {

                    /* This packet has no frame headers in it */

                    if (url_ftell(pb) >= ffm->packet_size * 3) {

                        url_fseek(pb, -ffm->packet_size * 2, SEEK_CUR);

                        goto retry_read;

                    }

                    /* This is bad, we cannot find a valid frame header */

                    return 0;

                }

                ffm->first_packet = 0;

                if ((frame_offset & 0x7ffff) < FFM_HEADER_SIZE)

                    return -1;

                ffm->packet_ptr = ffm->packet + (frame_offset & 0x7fff) - FFM_HEADER_SIZE;

                if (!first)

                    break;

            } else {

                ffm->packet_ptr = ffm->packet;

            }

            goto redo;

        }

        memcpy(buf, ffm->packet_ptr, len);

        buf += len;

        ffm->packet_ptr += len;

        size -= len;

        first = 0;

    }

    return size1 - size;

}
