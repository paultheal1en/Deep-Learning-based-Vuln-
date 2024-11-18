void ff_http_auth_handle_header(HTTPAuthState *state, const char *key,

                                const char *value)

{

    if (!strcmp(key, "WWW-Authenticate") || !strcmp(key, "Proxy-Authenticate")) {

        const char *p;

        if (av_stristart(value, "Basic ", &p) &&

            state->auth_type <= HTTP_AUTH_BASIC) {

            state->auth_type = HTTP_AUTH_BASIC;

            state->realm[0] = 0;

            state->stale = 0;

            ff_parse_key_value(p, (ff_parse_key_val_cb) handle_basic_params,

                               state);

        } else if (av_stristart(value, "Digest ", &p) &&

                   state->auth_type <= HTTP_AUTH_DIGEST) {

            state->auth_type = HTTP_AUTH_DIGEST;

            memset(&state->digest_params, 0, sizeof(DigestParams));

            state->realm[0] = 0;

            state->stale = 0;

            ff_parse_key_value(p, (ff_parse_key_val_cb) handle_digest_params,

                               state);

            choose_qop(state->digest_params.qop,

                       sizeof(state->digest_params.qop));

            if (!av_strcasecmp(state->digest_params.stale, "true"))

                state->stale = 1;

        }

    } else if (!strcmp(key, "Authentication-Info")) {

        ff_parse_key_value(value, (ff_parse_key_val_cb) handle_digest_update,

                           state);

    }

}
