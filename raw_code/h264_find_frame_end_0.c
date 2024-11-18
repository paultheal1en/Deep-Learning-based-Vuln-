static int h264_find_frame_end(H264ParseContext *p, const uint8_t *buf,

                               int buf_size)

{

    int i;

    uint32_t state;

    ParseContext *pc = &p->pc;

//    mb_addr= pc->mb_addr - 1;

    state = pc->state;

    if (state > 13)

        state = 7;



    for (i = 0; i < buf_size; i++) {

        if (state == 7) {

            i += p->h264dsp.startcode_find_candidate(buf + i, buf_size - i);

            if (i < buf_size)

                state = 2;

        } else if (state <= 2) {

            if (buf[i] == 1)

                state ^= 5;            // 2->7, 1->4, 0->5

            else if (buf[i])

                state = 7;

            else

                state >>= 1;           // 2->1, 1->0, 0->0

        } else if (state <= 5) {

            int nalu_type = buf[i] & 0x1F;

            if (nalu_type == NAL_SEI || nalu_type == NAL_SPS ||

                nalu_type == NAL_PPS || nalu_type == NAL_AUD) {

                if (pc->frame_start_found) {

                    i++;

                    goto found;

                }

            } else if (nalu_type == NAL_SLICE || nalu_type == NAL_DPA ||

                       nalu_type == NAL_IDR_SLICE) {

                if (pc->frame_start_found) {

                    state += 8;

                    continue;

                } else

                    pc->frame_start_found = 1;

            }

            state = 7;

        } else {

            // first_mb_in_slice is 0, probably the first nal of a new slice

            if (buf[i] & 0x80)

                goto found;

            state = 7;

        }

    }

    pc->state = state;

    return END_NOT_FOUND;



found:

    pc->state             = 7;

    pc->frame_start_found = 0;

    return i - (state & 5);

}
