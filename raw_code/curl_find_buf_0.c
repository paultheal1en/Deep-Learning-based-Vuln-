static int curl_find_buf(BDRVCURLState *s, size_t start, size_t len,

                         CURLAIOCB *acb)

{

    int i;

    size_t end = start + len;



    for (i=0; i<CURL_NUM_STATES; i++) {

        CURLState *state = &s->states[i];

        size_t buf_end = (state->buf_start + state->buf_off);

        size_t buf_fend = (state->buf_start + state->buf_len);



        if (!state->orig_buf)

            continue;

        if (!state->buf_off)

            continue;



        // Does the existing buffer cover our section?

        if ((start >= state->buf_start) &&

            (start <= buf_end) &&

            (end >= state->buf_start) &&

            (end <= buf_end))

        {

            char *buf = state->orig_buf + (start - state->buf_start);



            qemu_iovec_from_buf(acb->qiov, 0, buf, len);

            acb->common.cb(acb->common.opaque, 0);



            return FIND_RET_OK;

        }



        // Wait for unfinished chunks

        if ((start >= state->buf_start) &&

            (start <= buf_fend) &&

            (end >= state->buf_start) &&

            (end <= buf_fend))

        {

            int j;



            acb->start = start - state->buf_start;

            acb->end = acb->start + len;



            for (j=0; j<CURL_NUM_ACB; j++) {

                if (!state->acb[j]) {

                    state->acb[j] = acb;

                    return FIND_RET_WAIT;

                }

            }

        }

    }



    return FIND_RET_NONE;

}
