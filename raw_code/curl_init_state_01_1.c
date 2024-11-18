static CURLState *curl_init_state(BlockDriverState *bs, BDRVCURLState *s)

{

    CURLState *state = NULL;

    int i, j;



    do {

        for (i=0; i<CURL_NUM_STATES; i++) {

            for (j=0; j<CURL_NUM_ACB; j++)

                if (s->states[i].acb[j])

                    continue;

            if (s->states[i].in_use)

                continue;



            state = &s->states[i];

            state->in_use = 1;

            break;

        }

        if (!state) {

            qemu_mutex_unlock(&s->mutex);

            aio_poll(bdrv_get_aio_context(bs), true);

            qemu_mutex_lock(&s->mutex);

        }

    } while(!state);



    if (!state->curl) {

        state->curl = curl_easy_init();

        if (!state->curl) {

            return NULL;

        }

        curl_easy_setopt(state->curl, CURLOPT_URL, s->url);

        curl_easy_setopt(state->curl, CURLOPT_SSL_VERIFYPEER,

                         (long) s->sslverify);

        if (s->cookie) {

            curl_easy_setopt(state->curl, CURLOPT_COOKIE, s->cookie);

        }

        curl_easy_setopt(state->curl, CURLOPT_TIMEOUT, (long)s->timeout);

        curl_easy_setopt(state->curl, CURLOPT_WRITEFUNCTION,

                         (void *)curl_read_cb);

        curl_easy_setopt(state->curl, CURLOPT_WRITEDATA, (void *)state);

        curl_easy_setopt(state->curl, CURLOPT_PRIVATE, (void *)state);

        curl_easy_setopt(state->curl, CURLOPT_AUTOREFERER, 1);

        curl_easy_setopt(state->curl, CURLOPT_FOLLOWLOCATION, 1);

        curl_easy_setopt(state->curl, CURLOPT_NOSIGNAL, 1);

        curl_easy_setopt(state->curl, CURLOPT_ERRORBUFFER, state->errmsg);

        curl_easy_setopt(state->curl, CURLOPT_FAILONERROR, 1);



        if (s->username) {

            curl_easy_setopt(state->curl, CURLOPT_USERNAME, s->username);

        }

        if (s->password) {

            curl_easy_setopt(state->curl, CURLOPT_PASSWORD, s->password);

        }

        if (s->proxyusername) {

            curl_easy_setopt(state->curl,

                             CURLOPT_PROXYUSERNAME, s->proxyusername);

        }

        if (s->proxypassword) {

            curl_easy_setopt(state->curl,

                             CURLOPT_PROXYPASSWORD, s->proxypassword);

        }



        /* Restrict supported protocols to avoid security issues in the more

         * obscure protocols.  For example, do not allow POP3/SMTP/IMAP see

         * CVE-2013-0249.

         *

         * Restricting protocols is only supported from 7.19.4 upwards.

         */

#if LIBCURL_VERSION_NUM >= 0x071304

        curl_easy_setopt(state->curl, CURLOPT_PROTOCOLS, PROTOCOLS);

        curl_easy_setopt(state->curl, CURLOPT_REDIR_PROTOCOLS, PROTOCOLS);

#endif



#ifdef DEBUG_VERBOSE

        curl_easy_setopt(state->curl, CURLOPT_VERBOSE, 1);

#endif

    }



    QLIST_INIT(&state->sockets);

    state->s = s;



    return state;

}
