static int get_number(void *obj, const char *name, const AVOption **o_out, double *num, int *den, int64_t *intnum)

{

    const AVOption *o = av_opt_find(obj, name, NULL, 0, 0);

    void *dst;

    if (!o || o->offset<=0)

        goto error;



    dst= ((uint8_t*)obj) + o->offset;



    if (o_out) *o_out= o;



    switch (o->type) {

    case FF_OPT_TYPE_FLAGS:     *intnum= *(unsigned int*)dst;return 0;

    case FF_OPT_TYPE_INT:       *intnum= *(int    *)dst;return 0;

    case FF_OPT_TYPE_INT64:     *intnum= *(int64_t*)dst;return 0;

    case FF_OPT_TYPE_FLOAT:     *num=    *(float  *)dst;return 0;

    case FF_OPT_TYPE_DOUBLE:    *num=    *(double *)dst;return 0;

    case FF_OPT_TYPE_RATIONAL:  *intnum= ((AVRational*)dst)->num;

                                *den   = ((AVRational*)dst)->den;

                                                        return 0;

    }

error:

    *den=*intnum=0;

    return -1;

}
