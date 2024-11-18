static int dirac_combine_frame(AVCodecParserContext *s, AVCodecContext *avctx,

                               int next, const uint8_t **buf, int *buf_size)

{

    int parse_timing_info = (s->pts == AV_NOPTS_VALUE &&

                             s->dts == AV_NOPTS_VALUE);

    DiracParseContext *pc = s->priv_data;



    if (pc->overread_index) {

        memmove(pc->buffer, pc->buffer + pc->overread_index,

               pc->index - pc->overread_index);

        pc->index         -= pc->overread_index;

        pc->overread_index = 0;

        if (*buf_size == 0 && pc->buffer[4] == 0x10) {

            *buf      = pc->buffer;

            *buf_size = pc->index;

            return 0;

        }

    }



    if (next == -1) {

        /* Found a possible frame start but not a frame end */

        void *new_buffer =

            av_fast_realloc(pc->buffer, &pc->buffer_size,

                            pc->index + (*buf_size - pc->sync_offset));

        if (!new_buffer)

            return AVERROR(ENOMEM);

        pc->buffer = new_buffer;

        memcpy(pc->buffer + pc->index, (*buf + pc->sync_offset),

               *buf_size - pc->sync_offset);

        pc->index += *buf_size - pc->sync_offset;

        return -1;

    } else {

        /* Found a possible frame start and a  possible frame end */

        DiracParseUnit pu1, pu;

        void *new_buffer = av_fast_realloc(pc->buffer, &pc->buffer_size,

                                           pc->index + next);

        if (!new_buffer)

            return AVERROR(ENOMEM);

        pc->buffer = new_buffer;

        memcpy(pc->buffer + pc->index, *buf, next);

        pc->index += next;



        /* Need to check if we have a valid Parse Unit. We can't go by the

         * sync pattern 'BBCD' alone because arithmetic coding of the residual

         * and motion data can cause the pattern triggering a false start of

         * frame. So check if the previous parse offset of the next parse unit

         * is equal to the next parse offset of the current parse unit then

         * we can be pretty sure that we have a valid parse unit */

        if (!unpack_parse_unit(&pu1, pc, pc->index - 13)                     ||

            !unpack_parse_unit(&pu, pc, pc->index - 13 - pu1.prev_pu_offset) ||

            pu.next_pu_offset != pu1.prev_pu_offset                          ||

            pc->index < pc->dirac_unit_size + 13LL + pu1.prev_pu_offset

        ) {

            pc->index              -= 9;

            *buf_size               = next - 9;

            pc->header_bytes_needed = 9;

            return -1;

        }



        /* All non-frame data must be accompanied by frame data. This is to

         * ensure that pts is set correctly. So if the current parse unit is

         * not frame data, wait for frame data to come along */



        pc->dirac_unit = pc->buffer + pc->index - 13 -

                         pu1.prev_pu_offset - pc->dirac_unit_size;



        pc->dirac_unit_size += pu.next_pu_offset;



        if ((pu.pu_type & 0x08) != 0x08) {

            pc->header_bytes_needed = 9;

            *buf_size               = next;

            return -1;

        }



        /* Get the picture number to set the pts and dts*/

        if (parse_timing_info) {

            uint8_t *cur_pu = pc->buffer +

                              pc->index - 13 - pu1.prev_pu_offset;

            int pts = AV_RB32(cur_pu + 13);

            if (s->last_pts == 0 && s->last_dts == 0)

                s->dts = pts - 1;

            else

                s->dts = s->last_dts + 1;

            s->pts = pts;

            if (!avctx->has_b_frames && (cur_pu[4] & 0x03))

                avctx->has_b_frames = 1;

        }

        if (avctx->has_b_frames && s->pts == s->dts)

            s->pict_type = AV_PICTURE_TYPE_B;



        /* Finally have a complete Dirac data unit */

        *buf      = pc->dirac_unit;

        *buf_size = pc->dirac_unit_size;



        pc->dirac_unit_size     = 0;

        pc->overread_index      = pc->index - 13;

        pc->header_bytes_needed = 9;

    }

    return next;

}
