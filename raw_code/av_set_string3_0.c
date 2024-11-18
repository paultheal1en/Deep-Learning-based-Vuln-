int av_set_string3(void *obj, const char *name, const char *val, int alloc, const AVOption **o_out)

{

    int ret;

    const AVOption *o = av_opt_find(obj, name, NULL, 0, 0);

    if (o_out)

        *o_out = o;

    if (!o)

        return AVERROR_OPTION_NOT_FOUND;

    if (!val || o->offset<=0)

        return AVERROR(EINVAL);



    if (o->type == FF_OPT_TYPE_BINARY) {

        uint8_t **dst = (uint8_t **)(((uint8_t*)obj) + o->offset);

        int *lendst = (int *)(dst + 1);

        uint8_t *bin, *ptr;

        int len = strlen(val);

        av_freep(dst);

        *lendst = 0;

        if (len & 1) return AVERROR(EINVAL);

        len /= 2;

        ptr = bin = av_malloc(len);

        while (*val) {

            int a = hexchar2int(*val++);

            int b = hexchar2int(*val++);

            if (a < 0 || b < 0) {

                av_free(bin);

                return AVERROR(EINVAL);

            }

            *ptr++ = (a << 4) | b;

        }

        *dst = bin;

        *lendst = len;

        return 0;

    }

    if (o->type != FF_OPT_TYPE_STRING) {

        int notfirst=0;

        for (;;) {

            int i;

            char buf[256];

            int cmd=0;

            double d;



            if (*val == '+' || *val == '-')

                cmd= *(val++);



            for (i=0; i<sizeof(buf)-1 && val[i] && val[i]!='+' && val[i]!='-'; i++)

                buf[i]= val[i];

            buf[i]=0;



            {

                const AVOption *o_named = av_opt_find(obj, buf, o->unit, 0, 0);

                if (o_named && o_named->type == FF_OPT_TYPE_CONST)

                    d= o_named->default_val.dbl;

                else if (!strcmp(buf, "default")) d= o->default_val.dbl;

                else if (!strcmp(buf, "max"    )) d= o->max;

                else if (!strcmp(buf, "min"    )) d= o->min;

                else if (!strcmp(buf, "none"   )) d= 0;

                else if (!strcmp(buf, "all"    )) d= ~0;

                else {

                    int res = av_expr_parse_and_eval(&d, buf, const_names, const_values, NULL, NULL, NULL, NULL, NULL, 0, obj);

                    if (res < 0) {

                        av_log(obj, AV_LOG_ERROR, "Unable to parse option value \"%s\"\n", val);

                        return res;

                    }

                }

            }

            if (o->type == FF_OPT_TYPE_FLAGS) {

                if      (cmd=='+') d= av_get_int(obj, name, NULL) | (int64_t)d;

                else if (cmd=='-') d= av_get_int(obj, name, NULL) &~(int64_t)d;

            } else {

                if      (cmd=='+') d= notfirst*av_get_double(obj, name, NULL) + d;

                else if (cmd=='-') d= notfirst*av_get_double(obj, name, NULL) - d;

            }



            if ((ret = av_set_number2(obj, name, d, 1, 1, o_out)) < 0)

                return ret;

            val+= i;

            if (!*val)

                return 0;

            notfirst=1;

        }

        return AVERROR(EINVAL);

    }



    if (alloc) {

        av_free(*(void**)(((uint8_t*)obj) + o->offset));

        val= av_strdup(val);

    }



    memcpy(((uint8_t*)obj) + o->offset, &val, sizeof(val));

    return 0;

}
