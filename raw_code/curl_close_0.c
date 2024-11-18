static void curl_close(BlockDriverState *bs)

{

    BDRVCURLState *s = bs->opaque;

    int i;



    DPRINTF("CURL: Close\n");

    for (i=0; i<CURL_NUM_STATES; i++) {

        if (s->states[i].in_use)

            curl_clean_state(&s->states[i]);

        if (s->states[i].curl) {

            curl_easy_cleanup(s->states[i].curl);

            s->states[i].curl = NULL;

        }

        if (s->states[i].orig_buf) {

            g_free(s->states[i].orig_buf);

            s->states[i].orig_buf = NULL;

        }

    }

    if (s->multi)

        curl_multi_cleanup(s->multi);



    timer_del(&s->timer);



    g_free(s->url);

}
