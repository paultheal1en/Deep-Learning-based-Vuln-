static void save_bits(WMAProDecodeCtx *s, GetBitContext* gb, int len,
                      int append)
{
    int buflen;
    /** when the frame data does not need to be concatenated, the input buffer
        is resetted and additional bits from the previous frame are copyed
        and skipped later so that a fast byte copy is possible */
    if (!append) {
        s->frame_offset = get_bits_count(gb) & 7;
        s->num_saved_bits = s->frame_offset;
        init_put_bits(&s->pb, s->frame_data, MAX_FRAMESIZE);
    buflen = (s->num_saved_bits + len + 8) >> 3;
    if (len <= 0 || buflen > MAX_FRAMESIZE) {
        avpriv_request_sample(s->avctx, "Too small input buffer");
    s->num_saved_bits += len;
    if (!append) {
        avpriv_copy_bits(&s->pb, gb->buffer + (get_bits_count(gb) >> 3),
                     s->num_saved_bits);
    } else {
        int align = 8 - (get_bits_count(gb) & 7);
        align = FFMIN(align, len);
        put_bits(&s->pb, align, get_bits(gb, align));
        len -= align;
        avpriv_copy_bits(&s->pb, gb->buffer + (get_bits_count(gb) >> 3), len);
    skip_bits_long(gb, len);
    {
        PutBitContext tmp = s->pb;
        flush_put_bits(&tmp);
    init_get_bits(&s->gb, s->frame_data, s->num_saved_bits);
    skip_bits(&s->gb, s->frame_offset);