const char *av_get_string(void *obj, const char *name, const AVOption **o_out, char *buf, int buf_len)

{

    const AVOption *o = av_opt_find(obj, name, NULL, 0, 0);

    void *dst;

    uint8_t *bin;

    int len, i;

    if (!o || o->offset<=0)

        return NULL;

    if (o->type != FF_OPT_TYPE_STRING && (!buf || !buf_len))

        return NULL;



    dst= ((uint8_t*)obj) + o->offset;

    if (o_out) *o_out= o;



    switch (o->type) {

    case FF_OPT_TYPE_FLAGS:     snprintf(buf, buf_len, "0x%08X",*(int    *)dst);break;

    case FF_OPT_TYPE_INT:       snprintf(buf, buf_len, "%d" , *(int    *)dst);break;

    case FF_OPT_TYPE_INT64:     snprintf(buf, buf_len, "%"PRId64, *(int64_t*)dst);break;

    case FF_OPT_TYPE_FLOAT:     snprintf(buf, buf_len, "%f" , *(float  *)dst);break;

    case FF_OPT_TYPE_DOUBLE:    snprintf(buf, buf_len, "%f" , *(double *)dst);break;

    case FF_OPT_TYPE_RATIONAL:  snprintf(buf, buf_len, "%d/%d", ((AVRational*)dst)->num, ((AVRational*)dst)->den);break;

    case FF_OPT_TYPE_STRING:    return *(void**)dst;

    case FF_OPT_TYPE_BINARY:

        len = *(int*)(((uint8_t *)dst) + sizeof(uint8_t *));

        if (len >= (buf_len + 1)/2) return NULL;

        bin = *(uint8_t**)dst;

        for (i = 0; i < len; i++) snprintf(buf + i*2, 3, "%02X", bin[i]);

        break;

    default: return NULL;

    }

    return buf;

}
