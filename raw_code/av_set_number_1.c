static const AVOption *av_set_number(void *obj, const char *name, double num, int den, int64_t intnum){

    const AVOption *o= av_find_opt(obj, name, NULL, 0, 0);

    void *dst;

    if(!o || o->offset<=0)

        return NULL;



    if(o->max*den < num*intnum || o->min*den > num*intnum) {

        av_log(NULL, AV_LOG_ERROR, "Value %lf for parameter '%s' out of range\n", num, name);

        return NULL;

    }



    dst= ((uint8_t*)obj) + o->offset;



    switch(o->type){

    case FF_OPT_TYPE_FLAGS:

    case FF_OPT_TYPE_INT:   *(int       *)dst= llrint(num/den)*intnum; break;

    case FF_OPT_TYPE_INT64: *(int64_t   *)dst= llrint(num/den)*intnum; break;

    case FF_OPT_TYPE_FLOAT: *(float     *)dst= num*intnum/den;         break;

    case FF_OPT_TYPE_DOUBLE:*(double    *)dst= num*intnum/den;         break;

    case FF_OPT_TYPE_RATIONAL:

        if((int)num == num) *(AVRational*)dst= (AVRational){num*intnum, den};

        else                *(AVRational*)dst= av_d2q(num*intnum/den, 1<<24);

        break;

    default:

        return NULL;

    }

    return o;

}
