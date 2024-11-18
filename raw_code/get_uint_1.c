static inline unsigned int get_uint(ShortenContext *s, int k)

{

    if (s->version != 0)

        k = get_ur_golomb_shorten(&s->gb, ULONGSIZE);

    return get_ur_golomb_shorten(&s->gb, k);

}
