static int dca_find_frame_end(DCAParseContext * pc1, const uint8_t * buf,

                              int buf_size)

{

    int start_found, i;

    uint32_t state;

    ParseContext *pc = &pc1->pc;



    start_found = pc->frame_start_found;

    state = pc->state;



    i = 0;

    if (!start_found) {

        for (i = 0; i < buf_size; i++) {

            state = (state << 8) | buf[i];

            if (IS_MARKER(state, i, buf, buf_size)) {

                if (!pc1->lastmarker || state == pc1->lastmarker || pc1->lastmarker == DCA_HD_MARKER) {

                    start_found = 1;

                    pc1->lastmarker = state;

                    break;

                }

            }

        }

    }

    if (start_found) {

        for (; i < buf_size; i++) {

            pc1->size++;

            state = (state << 8) | buf[i];

            if (state == DCA_HD_MARKER && !pc1->hd_pos)

                pc1->hd_pos = pc1->size;

            if (IS_MARKER(state, i, buf, buf_size) && (state == pc1->lastmarker || pc1->lastmarker == DCA_HD_MARKER)) {

                if(pc1->framesize > pc1->size)

                    continue;

                if(!pc1->framesize){

                    pc1->framesize = pc1->hd_pos ? pc1->hd_pos : pc1->size;

                }

                pc->frame_start_found = 0;

                pc->state = -1;

                pc1->size = 0;

                return i - 3;

            }

        }

    }

    pc->frame_start_found = start_found;

    pc->state = state;

    return END_NOT_FOUND;

}
