static KeyValue *copy_key_value(KeyValue *src)

{

    KeyValue *dst = g_new(KeyValue, 1);

    memcpy(dst, src, sizeof(*src));






    return dst;
