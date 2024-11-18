AVOption *av_set_string(void *obj, const char *name, const char *val){

    AVOption *o= find_opt(obj, name);

    if(!o || !val || o->offset<=0) 

        return NULL;

    if(o->type != FF_OPT_TYPE_STRING){

        double d=0, tmp_d;

        for(;;){

            int i;

            char buf[256], *tail;



            for(i=0; i<sizeof(buf)-1 && val[i] && val[i]!='+'; i++)

                buf[i]= val[i];

            buf[i]=0;

            val+= i;

            

            tmp_d= av_parse_num(buf, &tail);

            if(tail > buf)

                d+= tmp_d;

            else{

                AVOption *o_named= find_opt(obj, buf);

                if(o_named && o_named->type == FF_OPT_TYPE_CONST) 

                    d+= o_named->default_val;

                else if(!strcmp(buf, "default")) d+= o->default_val;

                else if(!strcmp(buf, "max"    )) d+= o->max;

                else if(!strcmp(buf, "min"    )) d+= o->min;

                else return NULL;

            }



            if(*val == '+') val++;

            if(!*val)

                return av_set_number(obj, name, d, 1, 1);

        }

        return NULL;

    }

    

    memcpy(((uint8_t*)obj) + o->offset, val, sizeof(val));

    return o;

}
