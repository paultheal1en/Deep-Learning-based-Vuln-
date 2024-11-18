double av_get_double(void *obj, const char *name, const AVOption **o_out)

{

    int64_t intnum=1;

    double num=1;

    int den=1;



    av_get_number(obj, name, o_out, &num, &den, &intnum);

    return num*intnum/den;

}
