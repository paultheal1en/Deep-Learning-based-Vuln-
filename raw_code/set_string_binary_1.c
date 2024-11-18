static int set_string_binary(void *obj, const AVOption *o, const char *val, uint8_t **dst)

{

    int *lendst = (int *)(dst + 1);

    uint8_t *bin, *ptr;

    int len = strlen(val);



    av_freep(dst);

    *lendst = 0;



    if (len & 1)

        return AVERROR(EINVAL);

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
