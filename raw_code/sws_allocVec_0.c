SwsVector *sws_allocVec(int length)

{

    SwsVector *vec = av_malloc(sizeof(SwsVector));

    if (!vec)

        return NULL;

    vec->length = length;

    vec->coeff  = av_malloc(sizeof(double) * length);

    if (!vec->coeff)

        av_freep(&vec);

    return vec;

}
