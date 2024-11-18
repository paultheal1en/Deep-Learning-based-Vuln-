const AVOption *av_set_string(void *obj, const char *name, const char *val){

    const AVOption *o= av_find_opt(obj, name, NULL, 0, 0);

    if(o && o->offset==0 && o->type == FF_OPT_TYPE_CONST && o->unit){

        return set_all_opt(obj, o->unit, o->default_val);

    }

    if(!o || !val || o->offset<=0)

        return NULL;

    if(o->type != FF_OPT_TYPE_STRING){

        for(;;){

            int i;

            char buf[256];

            int cmd=0;

            double d;

            char *error = NULL;



            if(*val == '+' || *val == '-')

                cmd= *(val++);



            for(i=0; i<sizeof(buf)-1 && val[i] && val[i]!='+' && val[i]!='-'; i++)

                buf[i]= val[i];

            buf[i]=0;

            val+= i;



            d = ff_eval2(buf, const_values, const_names, NULL, NULL, NULL, NULL, NULL, &error);

            if(isnan(d)) {

                const AVOption *o_named= av_find_opt(obj, buf, o->unit, 0, 0);

                if(o_named && o_named->type == FF_OPT_TYPE_CONST)

                    d= o_named->default_val;

                else if(!strcmp(buf, "default")) d= o->default_val;

                else if(!strcmp(buf, "max"    )) d= o->max;

                else if(!strcmp(buf, "min"    )) d= o->min;

                else if(!strcmp(buf, "none"   )) d= 0;

                else if(!strcmp(buf, "all"    )) d= ~0;

                else {

                    if (!error)

                        av_log(NULL, AV_LOG_ERROR, "Unable to parse option value \"%s\": %s\n", val, error);

                    return NULL;

                }

            }

            if(o->type == FF_OPT_TYPE_FLAGS){

                if     (cmd=='+') d= av_get_int(obj, name, NULL) | (int64_t)d;

                else if(cmd=='-') d= av_get_int(obj, name, NULL) &~(int64_t)d;

            }else if(cmd=='-')

                d= -d;



            av_set_number(obj, name, d, 1, 1);

            if(!*val)

                return o;

        }

        return NULL;

    }



    memcpy(((uint8_t*)obj) + o->offset, val, sizeof(val));

    return o;

}
