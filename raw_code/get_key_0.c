static int get_key(const char **ropts, const char *delim, char *key, unsigned key_size)

{

    unsigned key_pos = 0;

    const char *opts = *ropts;



    opts += strspn(opts, WHITESPACES);

    while (is_key_char(*opts)) {

        key[key_pos++] = *opts;

        if (key_pos == key_size)

            key_pos--;

        (opts)++;

    }

    opts += strspn(opts, WHITESPACES);

    if (!*opts || !strchr(delim, *opts))

        return AVERROR(EINVAL);

    opts++;

    key[key_pos++] = 0;

    if (key_pos == key_size)

        key[key_pos - 4] = key[key_pos - 3] = key[key_pos - 2] = '.';

    *ropts = opts;

    return 0;

}
