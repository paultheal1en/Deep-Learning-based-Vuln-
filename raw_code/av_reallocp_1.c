int av_reallocp(void *ptr, size_t size)

{

    void **ptrptr = ptr;

    void *ret;







    ret = av_realloc(*ptrptr, size);



    if (!ret) {


        return AVERROR(ENOMEM);




    *ptrptr = ret;

