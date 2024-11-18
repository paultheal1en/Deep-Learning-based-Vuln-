const char *av_get_string(void *obj, const char *name, const AVOption **o_out, char *buf, int buf_len){

    const AVOption *o= av_find_opt(obj, name, NULL, 0, 0);

    void *dst;

    if(!o || o->offset<=0)

        return NULL;

    if(o->type != FF_OPT_TYPE_STRING && (!buf || !buf_len))

        return NULL;



    dst= ((uint8_t*)obj) + o->offset;

    if(o_out) *o_out= o;



    if(o->type == FF_OPT_TYPE_STRING)

        return dst;



    switch(o->type){

    case FF_OPT_TYPE_FLAGS:     snprintf(buf, buf_len, "0x%08X",*(int    *)dst);break;

    case FF_OPT_TYPE_INT:       snprintf(buf, buf_len, "%d" , *(int    *)dst);break;

    case FF_OPT_TYPE_INT64:     snprintf(buf, buf_len, "%"PRId64, *(int64_t*)dst);break;

    case FF_OPT_TYPE_FLOAT:     snprintf(buf, buf_len, "%f" , *(float  *)dst);break;

    case FF_OPT_TYPE_DOUBLE:    snprintf(buf, buf_len, "%f" , *(double *)dst);break;

    case FF_OPT_TYPE_RATIONAL:  snprintf(buf, buf_len, "%d/%d", ((AVRational*)dst)->num, ((AVRational*)dst)->den);break;

    default: return NULL;

    }

    return buf;

}
