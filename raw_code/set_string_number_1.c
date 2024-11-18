static int set_string_number(void *obj, void *target_obj, const AVOption *o, const char *val, void *dst)

{

    int ret = 0;

    int num, den;

    char c;



    if (sscanf(val, "%d%*1[:/]%d%c", &num, &den, &c) == 2) {

        if ((ret = write_number(obj, o, dst, 1, den, num)) >= 0)

            return ret;

        ret = 0;

    }



    for (;;) {

        int i = 0;

        char buf[256];

        int cmd = 0;

        double d;

        int64_t intnum = 1;



        if (o->type == AV_OPT_TYPE_FLAGS) {

            if (*val == '+' || *val == '-')

                cmd = *(val++);

            for (; i < sizeof(buf) - 1 && val[i] && val[i] != '+' && val[i] != '-'; i++)

                buf[i] = val[i];

            buf[i] = 0;

        }



        {

            const AVOption *o_named = av_opt_find(target_obj, buf, o->unit, 0, 0);

            int res;

            int ci = 0;

            double const_values[64];

            const char * const_names[64];

            if (o_named && o_named->type == AV_OPT_TYPE_CONST)

                d = DEFAULT_NUMVAL(o_named);

            else {

                if (o->unit) {

                    for (o_named = NULL; o_named = av_opt_next(target_obj, o_named); ) {

                        if (o_named->type == AV_OPT_TYPE_CONST &&

                            o_named->unit &&

                            !strcmp(o_named->unit, o->unit)) {

                            if (ci + 6 >= FF_ARRAY_ELEMS(const_values)) {

                                av_log(obj, AV_LOG_ERROR, "const_values array too small for %s\n", o->unit);

                                return AVERROR_PATCHWELCOME;

                            }

                            const_names [ci  ] = o_named->name;

                            const_values[ci++] = DEFAULT_NUMVAL(o_named);

                        }

                    }

                }

                const_names [ci  ] = "default";

                const_values[ci++] = DEFAULT_NUMVAL(o);

                const_names [ci  ] = "max";

                const_values[ci++] = o->max;

                const_names [ci  ] = "min";

                const_values[ci++] = o->min;

                const_names [ci  ] = "none";

                const_values[ci++] = 0;

                const_names [ci  ] = "all";

                const_values[ci++] = ~0;

                const_names [ci] = NULL;

                const_values[ci] = 0;



                res = av_expr_parse_and_eval(&d, i ? buf : val, const_names,

                                            const_values, NULL, NULL, NULL, NULL, NULL, 0, obj);

                if (res < 0) {

                    av_log(obj, AV_LOG_ERROR, "Unable to parse option value \"%s\"\n", val);

                    return res;

                }

            }

        }

        if (o->type == AV_OPT_TYPE_FLAGS) {

            read_number(o, dst, NULL, NULL, &intnum);

            if      (cmd == '+') d = intnum | (int64_t)d;

            else if (cmd == '-') d = intnum &~(int64_t)d;

        }



        if ((ret = write_number(obj, o, dst, d, 1, 1)) < 0)

            return ret;

        val += i;

        if (!i || !*val)

            return 0;

    }



    return 0;

}
