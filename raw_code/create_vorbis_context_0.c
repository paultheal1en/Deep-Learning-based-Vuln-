static void create_vorbis_context(venc_context_t * venc, AVCodecContext * avccontext) {

    codebook_t * cb;

    floor_t * fc;

    residue_t * rc;

    mapping_t * mc;

    int i, book;



    venc->channels = avccontext->channels;

    venc->sample_rate = avccontext->sample_rate;

    venc->blocksize[0] = venc->blocksize[1] = 11;



    venc->ncodebooks = 29;

    venc->codebooks = av_malloc(sizeof(codebook_t) * venc->ncodebooks);



    int codebook0[] = { 2, 10, 8, 14, 7, 12, 11, 14, 1, 5, 3, 7, 4, 9, 7, 13, };

    int codebook1[] = { 1, 4, 2, 6, 3, 7, 5, 7, };

    int codebook2[] = { 1, 5, 7, 21, 5, 8, 9, 21, 10, 9, 12, 20, 20, 16, 20, 20, 4, 8, 9, 20, 6, 8, 9, 20, 11, 11, 13, 20, 20, 15, 17, 20, 9, 11, 14, 20, 8, 10, 15, 20, 11, 13, 15, 20, 20, 20, 20, 20, 20, 20, 20, 20, 13, 20, 20, 20, 18, 18, 20, 20, 20, 20, 20, 20, 3, 6, 8, 20, 6, 7, 9, 20, 10, 9, 12, 20, 20, 20, 20, 20, 5, 7, 9, 20, 6, 6, 9, 20, 10, 9, 12, 20, 20, 20, 20, 20, 8, 10, 13, 20, 8, 9, 12, 20, 11, 10, 12, 20, 20, 20, 20, 20, 18, 20, 20, 20, 15, 17, 18, 20, 18, 17, 18, 20, 20, 20, 20, 20, 7, 10, 12, 20, 8, 9, 11, 20, 14, 13, 14, 20, 20, 20, 20, 20, 6, 9, 12, 20, 7, 8, 11, 20, 12, 11, 13, 20, 20, 20, 20, 20, 9, 11, 15, 20, 8, 10, 14, 20, 12, 11, 14, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 11, 16, 18, 20, 15, 15, 17, 20, 20, 17, 20, 20, 20, 20, 20, 20, 9, 14, 16, 20, 12, 12, 15, 20, 17, 15, 18, 20, 20, 20, 20, 20, 16, 19, 18, 20, 15, 16, 20, 20, 17, 17, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, };

    int codebook3[] = { 2, 3, 7, 13, 4, 4, 7, 15, 8, 6, 9, 17, 21, 16, 15, 21, 2, 5, 7, 11, 5, 5, 7, 14, 9, 7, 10, 16, 17, 15, 16, 21, 4, 7, 10, 17, 7, 7, 9, 15, 11, 9, 11, 16, 21, 18, 15, 21, 18, 21, 21, 21, 15, 17, 17, 19, 21, 19, 18, 20, 21, 21, 21, 20, };

    int codebook4[] = { 5, 5, 5, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 7, 5, 7, 5, 7, 5, 7, 5, 8, 6, 8, 6, 8, 6, 9, 6, 9, 6, 10, 6, 10, 6, 11, 6, 11, 7, 11, 7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 7, 12, 8, 13, 8, 12, 8, 12, 8, 13, 8, 13, 9, 13, 9, 13, 9, 13, 9, 12, 10, 12, 10, 13, 10, 14, 11, 14, 12, 14, 13, 14, 13, 14, 14, 15, 16, 15, 15, 15, 14, 15, 17, 21, 22, 22, 21, 22, 22, 22, 22, 22, 22, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, };

    int codebook5[] = { 2, 5, 5, 4, 5, 4, 5, 4, 5, 4, 6, 5, 6, 5, 6, 5, 6, 5, 7, 5, 7, 6, 8, 6, 8, 6, 8, 6, 9, 6, 9, 6, };

    int codebook6[] = { 8, 5, 8, 4, 9, 4, 9, 4, 9, 4, 9, 4, 9, 4, 9, 4, 9, 4, 9, 4, 9, 4, 8, 4, 8, 4, 9, 5, 9, 5, 9, 5, 9, 5, 9, 6, 10, 6, 10, 7, 10, 8, 11, 9, 11, 11, 12, 13, 12, 14, 13, 15, 13, 15, 14, 16, 14, 17, 15, 17, 15, 15, 16, 16, 15, 16, 16, 16, 15, 18, 16, 15, 17, 17, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, };

    int codebook7[] = { 1, 5, 5, 5, 5, 5, 5, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 6, 7, 7, 7, 7, 8, 7, 8, 8, 9, 8, 10, 9, 10, 9, };

    int codebook8[] = { 4, 3, 4, 3, 4, 4, 5, 4, 5, 4, 5, 5, 6, 5, 6, 5, 7, 5, 7, 6, 7, 6, 8, 7, 8, 7, 8, 7, 9, 8, 9, 9, 9, 9, 10, 10, 10, 11, 9, 12, 9, 12, 9, 15, 10, 14, 9, 13, 10, 13, 10, 12, 10, 12, 10, 13, 10, 12, 11, 13, 11, 14, 12, 13, 13, 14, 14, 13, 14, 15, 14, 16, 13, 13, 14, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 15, 15, };

    int codebook9[] = { 4, 5, 4, 5, 3, 5, 3, 5, 3, 5, 4, 4, 4, 4, 5, 5, 5, };

    int codebook10[] = { 3, 3, 4, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 5, 7, 5, 8, 6, 8, 6, 9, 7, 10, 7, 10, 8, 10, 8, 11, 9, 11, };

    int codebook11[] = { 3, 7, 3, 8, 3, 10, 3, 8, 3, 9, 3, 8, 4, 9, 4, 9, 5, 9, 6, 10, 6, 9, 7, 11, 7, 12, 9, 13, 10, 13, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, };

    int codebook12[] = { 4, 5, 4, 5, 4, 5, 4, 5, 3, 5, 3, 5, 3, 5, 4, 5, 4, };

    int codebook13[] = { 4, 2, 4, 2, 5, 3, 5, 4, 6, 6, 6, 7, 7, 8, 7, 8, 7, 8, 7, 9, 8, 9, 8, 9, 8, 10, 8, 11, 9, 12, 9, 12, };

    int codebook14[] = { 2, 5, 2, 6, 3, 6, 4, 7, 4, 7, 5, 9, 5, 11, 6, 11, 6, 11, 7, 11, 6, 11, 6, 11, 9, 11, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 10, 10, 10, 10, 10, 10, };

    int codebook15[] = { 5, 6, 11, 11, 11, 11, 10, 10, 12, 11, 5, 2, 11, 5, 6, 6, 7, 9, 11, 13, 13, 10, 7, 11, 6, 7, 8, 9, 10, 12, 11, 5, 11, 6, 8, 7, 9, 11, 14, 15, 11, 6, 6, 8, 4, 5, 7, 8, 10, 13, 10, 5, 7, 7, 5, 5, 6, 8, 10, 11, 10, 7, 7, 8, 6, 5, 5, 7, 9, 9, 11, 8, 8, 11, 8, 7, 6, 6, 7, 9, 12, 11, 10, 13, 9, 9, 7, 7, 7, 9, 11, 13, 12, 15, 12, 11, 9, 8, 8, 8, };

    int codebook16[] = { 2, 4, 4, 0, 0, 0, 0, 0, 0, 5, 6, 6, 0, 0, 0, 0, 0, 0, 5, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 7, 7, 0, 0, 0, 0, 0, 0, 7, 8, 8, 0, 0, 0, 0, 0, 0, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 7, 7, 0, 0, 0, 0, 0, 0, 6, 8, 7, 0, 0, 0, 0, 0, 0, 7, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 7, 7, 0, 0, 0, 0, 0, 0, 7, 8, 8, 0, 0, 0, 0, 0, 0, 7, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 8, 0, 0, 0, 0, 0, 0, 8, 8, 9, 0, 0, 0, 0, 0, 0, 8, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 8, 8, 0, 0, 0, 0, 0, 0, 7, 9, 8, 0, 0, 0, 0, 0, 0, 8, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 7, 7, 0, 0, 0, 0, 0, 0, 7, 8, 8, 0, 0, 0, 0, 0, 0, 7, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 8, 8, 0, 0, 0, 0, 0, 0, 8, 9, 9, 0, 0, 0, 0, 0, 0, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 8, 8, 0, 0, 0, 0, 0, 0, 8, 9, 9, 0, 0, 0, 0, 0, 0, 8, 9, 8, };

    int codebook17[] = { 2, 5, 5, 0, 0, 0, 5, 5, 0, 0, 0, 5, 5, 0, 0, 0, 7, 8, 0, 0, 0, 0, 0, 0, 0, 5, 6, 6, 0, 0, 0, 7, 7, 0, 0, 0, 7, 7, 0, 0, 0, 10, 10, 0, 0, 0, 0, 0, 0, 0, 5, 6, 6, 0, 0, 0, 7, 7, 0, 0, 0, 7, 7, 0, 0, 0, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 7, 7, 0, 0, 0, 7, 7, 0, 0, 0, 7, 7, 0, 0, 0, 9, 9, 0, 0, 0, 0, 0, 0, 0, 5, 7, 7, 0, 0, 0, 7, 7, 0, 0, 0, 7, 7, 0, 0, 0, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 7, 7, 0, 0, 0, 7, 7, 0, 0, 0, 7, 7, 0, 0, 0, 9, 9, 0, 0, 0, 0, 0, 0, 0, 5, 7, 7, 0, 0, 0, 7, 7, 0, 0, 0, 7, 7, 0, 0, 0, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 10, 0, 0, 0, 9, 9, 0, 0, 0, 9, 9, 0, 0, 0, 10, 10, 0, 0, 0, 0, 0, 0, 0, 8, 10, 10, 0, 0, 0, 9, 9, 0, 0, 0, 9, 9, 0, 0, 0, 10, 10, };

    int codebook18[] = { 2, 4, 3, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 6, 6, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 6, 7, 9, 9, };

    int codebook19[] = { 2, 3, 3, 6, 6, 0, 0, 0, 0, 0, 4, 4, 6, 6, 0, 0, 0, 0, 0, 4, 4, 6, 6, 0, 0, 0, 0, 0, 5, 5, 6, 6, 0, 0, 0, 0, 0, 0, 0, 6, 6, 0, 0, 0, 0, 0, 0, 0, 7, 8, 0, 0, 0, 0, 0, 0, 0, 7, 7, 0, 0, 0, 0, 0, 0, 0, 9, 9, };

    int codebook20[] = { 1, 3, 4, 6, 6, 7, 7, 9, 9, 0, 5, 5, 7, 7, 7, 8, 9, 9, 0, 5, 5, 7, 7, 8, 8, 9, 9, 0, 7, 7, 8, 8, 8, 8, 10, 10, 0, 0, 0, 8, 8, 8, 8, 10, 10, 0, 0, 0, 9, 9, 9, 9, 10, 10, 0, 0, 0, 9, 9, 9, 9, 10, 10, 0, 0, 0, 10, 10, 10, 10, 11, 11, 0, 0, 0, 0, 0, 10, 10, 11, 11, };

    int codebook21[] = { 2, 3, 3, 6, 6, 7, 7, 8, 8, 8, 8, 9, 9, 10, 10, 11, 10, 0, 5, 5, 7, 7, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 0, 5, 5, 7, 7, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 0, 6, 6, 7, 7, 8, 8, 9, 9, 9, 9, 10, 10, 11, 11, 11, 11, 0, 0, 0, 7, 7, 8, 8, 9, 9, 9, 9, 10, 10, 11, 11, 11, 12, 0, 0, 0, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 11, 11, 12, 12, 0, 0, 0, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 11, 11, 12, 12, 0, 0, 0, 9, 9, 9, 9, 10, 10, 10, 10, 11, 10, 11, 11, 12, 12, 0, 0, 0, 0, 0, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 0, 0, 0, 0, 0, 9, 8, 9, 9, 10, 10, 11, 11, 12, 12, 12, 12, 0, 0, 0, 0, 0, 8, 8, 9, 9, 10, 10, 11, 11, 12, 11, 12, 12, 0, 0, 0, 0, 0, 9, 10, 10, 10, 11, 11, 11, 11, 12, 12, 13, 13, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 11, 11, 12, 12, 13, 13, 0, 0, 0, 0, 0, 0, 0, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 0, 0, 0, 0, 0, 0, 0, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 0, 0, 0, 0, 0, 0, 0, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 12, 12, 12, 13, 13, 13, 13, };

    int codebook22[] = { 1, 4, 4, 7, 6, 6, 7, 6, 6, 4, 7, 7, 10, 9, 9, 11, 9, 9, 4, 7, 7, 10, 9, 9, 11, 9, 9, 7, 10, 10, 11, 11, 10, 12, 11, 11, 6, 9, 9, 11, 10, 10, 11, 10, 10, 6, 9, 9, 11, 10, 10, 11, 10, 10, 7, 11, 11, 11, 11, 11, 12, 11, 11, 6, 9, 9, 11, 10, 10, 11, 10, 10, 6, 9, 9, 11, 10, 10, 11, 10, 10, };

    int codebook23[] = { 2, 4, 4, 6, 6, 7, 7, 7, 7, 8, 8, 10, 5, 5, 6, 6, 7, 7, 8, 8, 8, 8, 10, 5, 5, 6, 6, 7, 7, 8, 8, 8, 8, 10, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 10, 10, 10, 7, 7, 8, 7, 8, 8, 8, 8, 10, 10, 10, 8, 8, 8, 8, 8, 8, 8, 8, 10, 10, 10, 7, 8, 8, 8, 8, 8, 8, 8, 10, 10, 10, 8, 8, 8, 8, 8, 8, 8, 8, 10, 10, 10, 10, 10, 8, 8, 8, 8, 8, 8, 10, 10, 10, 10, 10, 9, 9, 8, 8, 9, 8, 10, 10, 10, 10, 10, 8, 8, 8, 8, 8, 8, };

    int codebook24[] = { 1, 4, 4, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 6, 5, 5, 7, 7, 8, 8, 8, 8, 9, 9, 10, 10, 7, 5, 5, 7, 7, 8, 8, 8, 8, 9, 9, 11, 10, 0, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 11, 11, 0, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 11, 11, 0, 12, 12, 9, 9, 10, 10, 10, 10, 11, 11, 11, 12, 0, 13, 13, 9, 9, 10, 10, 10, 10, 11, 11, 12, 12, 0, 0, 0, 10, 10, 10, 10, 11, 11, 12, 12, 12, 12, 0, 0, 0, 10, 10, 10, 10, 11, 11, 12, 12, 12, 12, 0, 0, 0, 14, 14, 11, 11, 11, 11, 12, 12, 13, 13, 0, 0, 0, 14, 14, 11, 11, 11, 11, 12, 12, 13, 13, 0, 0, 0, 0, 0, 12, 12, 12, 12, 13, 13, 14, 13, 0, 0, 0, 0, 0, 13, 13, 12, 12, 13, 12, 14, 13, };

    int codebook25[] = { 2, 4, 4, 5, 5, 6, 5, 5, 5, 5, 6, 4, 5, 5, 5, 6, 5, 5, 5, 5, 6, 6, 6, 5, 5, };

    int codebook26[] = { 1, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 4, 9, 8, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 2, 9, 7, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, };

    int codebook27[] = { 1, 4, 4, 6, 6, 7, 7, 8, 7, 9, 9, 10, 10, 10, 10, 6, 5, 5, 7, 7, 8, 8, 10, 8, 11, 10, 12, 12, 13, 13, 6, 5, 5, 7, 7, 8, 8, 10, 9, 11, 11, 12, 12, 13, 12, 18, 8, 8, 8, 8, 9, 9, 10, 9, 11, 10, 12, 12, 13, 13, 18, 8, 8, 8, 8, 9, 9, 10, 10, 11, 11, 13, 12, 14, 13, 18, 11, 11, 9, 9, 10, 10, 11, 11, 11, 12, 13, 12, 13, 14, 18, 11, 11, 9, 8, 11, 10, 11, 11, 11, 11, 12, 12, 14, 13, 18, 18, 18, 10, 11, 10, 11, 12, 12, 12, 12, 13, 12, 14, 13, 18, 18, 18, 10, 11, 11, 9, 12, 11, 12, 12, 12, 13, 13, 13, 18, 18, 17, 14, 14, 11, 11, 12, 12, 13, 12, 14, 12, 14, 13, 18, 18, 18, 14, 14, 11, 10, 12, 9, 12, 13, 13, 13, 13, 13, 18, 18, 17, 16, 18, 13, 13, 12, 12, 13, 11, 14, 12, 14, 14, 17, 18, 18, 17, 18, 13, 12, 13, 10, 12, 11, 14, 14, 14, 14, 17, 18, 18, 18, 18, 15, 16, 12, 12, 13, 10, 14, 12, 14, 15, 18, 18, 18, 16, 17, 16, 14, 12, 11, 13, 10, 13, 13, 14, 15, };

    int codebook28[] = { 2, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 10, 6, 6, 7, 7, 8, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 7, 7, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 11, 11, 11, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 9, 10, 10, 10, 11, 11, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 11, 10, 11, 11, 11, 9, 9, 9, 9, 9, 9, 10, 10, 9, 9, 10, 9, 11, 10, 11, 11, 11, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 9, 11, 11, 11, 11, 11, 9, 9, 9, 9, 10, 10, 9, 9, 9, 9, 10, 9, 11, 11, 11, 11, 11, 11, 11, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 10, 9, 10, 10, 9, 10, 9, 9, 10, 9, 11, 10, 10, 11, 11, 11, 11, 9, 10, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 10, 10, 10, 9, 9, 10, 9, 10, 9, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 9, 9, 9, 9, 9, 10, 10, 10, };



    int codebook_sizes[] = { 16, 8, 256, 64, 128, 32, 96, 32, 96, 17, 32, 78, 17, 32, 78, 100, 1641, 443, 105, 68, 81, 289, 81, 121, 169, 25, 169, 225, 289, };

    int * codebook_lens[] = { codebook0,  codebook1,  codebook2,  codebook3,  codebook4,  codebook5,  codebook6,  codebook7,

                              codebook8,  codebook9,  codebook10, codebook11, codebook12, codebook13, codebook14, codebook15,

                              codebook16, codebook17, codebook18, codebook19, codebook20, codebook21, codebook22, codebook23,

                              codebook24, codebook25, codebook26, codebook27, codebook28, };



    struct {

        int lookup;

        int dim;

        float min;

        float delta;

        int real_len;

        int * quant;

    } cvectors[] = {

        { 1, 8,    -1.0,   1.0, 6561,(int[]){ 1, 0, 2, } },

        { 1, 4,    -2.0,   1.0, 625, (int[]){ 2, 1, 3, 0, 4, } },

        { 1, 4,    -2.0,   1.0, 625, (int[]){ 2, 1, 3, 0, 4, } },

        { 1, 2,    -4.0,   1.0, 81,  (int[]){ 4, 3, 5, 2, 6, 1, 7, 0, 8, } },

        { 1, 2,    -4.0,   1.0, 81,  (int[]){ 4, 3, 5, 2, 6, 1, 7, 0, 8, } },

        { 1, 2,    -8.0,   1.0, 289, (int[]){ 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15, 0, 16, } },

        { 1, 4,   -11.0,  11.0, 81,  (int[]){ 1, 0, 2, } },

        { 1, 2,    -5.0,   1.0, 121, (int[]){ 5, 4, 6, 3, 7, 2, 8, 1, 9, 0, 10, } },

        { 1, 2,   -30.0,   5.0, 169, (int[]){ 6, 5, 7, 4, 8, 3, 9, 2, 10, 1, 11, 0, 12, } },

        { 1, 2,    -2.0,   1.0, 25,  (int[]){ 2, 1, 3, 0, 4, } },

        { 1, 2, -1530.0, 255.0, 169, (int[]){ 6, 5, 7, 4, 8, 3, 9, 2, 10, 1, 11, 0, 12, } },

        { 1, 2,  -119.0,  17.0, 225, (int[]){ 7, 6, 8, 5, 9, 4, 10, 3, 11, 2, 12, 1, 13, 0, 14, } },

        { 1, 2,    -8.0,   1.0, 289, (int[]){ 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15, 0, 16, } },

    };



    // codebook 0..14 - floor1 book, values 0..255

    // codebook 15 residue masterbook

    // codebook 16..29 residue

    for (book = 0; book < venc->ncodebooks; book++) {

        cb = &venc->codebooks[book];

        cb->nentries = codebook_sizes[book];

        if (book < 16) {

            cb->ndimentions = 2;

            cb->min = 0.;

            cb->delta = 0.;

            cb->seq_p = 0;

            cb->lookup = 0;

            cb->quantlist = NULL;

        } else {

            int vals;

            cb->seq_p = 0;

            cb->nentries = cvectors[book - 16].real_len;

            cb->ndimentions = cvectors[book - 16].dim;

            cb->min = cvectors[book - 16].min;

            cb->delta = cvectors[book - 16].delta;

            cb->lookup = cvectors[book - 16].lookup;

            vals = cb_lookup_vals(cb->lookup, cb->ndimentions, cb->nentries);

            cb->quantlist = av_malloc(sizeof(int) * vals);

            for (i = 0; i < vals; i++) cb->quantlist[i] = cvectors[book - 16].quant[i];

        }

        cb->entries = av_malloc(sizeof(cb_entry_t) * cb->nentries);

        for (i = 0; i < cb->nentries; i++) {

            if (i < codebook_sizes[book]) cb->entries[i].len = codebook_lens[book][i];

            else cb->entries[i].len = 0;

        }

        ready_codebook(cb);

    }



    venc->nfloors = 1;

    venc->floors = av_malloc(sizeof(floor_t) * venc->nfloors);



    // just 1 floor

    fc = &venc->floors[0];

    fc->partitions = 8;

    fc->partition_to_class = av_malloc(sizeof(int) * fc->partitions);

    fc->nclasses = 0;

    for (i = 0; i < fc->partitions; i++) {

        int a[] = {0,1,2,2,3,3,4,4};

        fc->partition_to_class[i] = a[i];

        fc->nclasses = FFMAX(fc->nclasses, fc->partition_to_class[i]);

    }

    fc->nclasses++;

    fc->classes = av_malloc(sizeof(floor_class_t) * fc->nclasses);

    for (i = 0; i < fc->nclasses; i++) {

        floor_class_t * c = &fc->classes[i];

        int j, books;

        int dim[] = {3,4,3,4,3};

        int subclass[] = {0,1,1,2,2};

        int masterbook[] = {0/*none*/,0,1,2,3};

        int * nbooks[] = {

            (int[]){ 4 },

            (int[]){ 5, 6 },

            (int[]){ 7, 8 },

            (int[]){ -1, 9, 10, 11 },

            (int[]){ -1, 12, 13, 14 },

        };

        c->dim = dim[i];

        c->subclass = subclass[i];

        c->masterbook = masterbook[i];

        books = (1 << c->subclass);

        c->books = av_malloc(sizeof(int) * books);

        for (j = 0; j < books; j++) c->books[j] = nbooks[i][j];

    }

    fc->multiplier = 2;

    fc->rangebits = venc->blocksize[0] - 1;



    fc->values = 2;

    for (i = 0; i < fc->partitions; i++)

        fc->values += fc->classes[fc->partition_to_class[i]].dim;



    fc->list = av_malloc(sizeof(floor_entry_t) * fc->values);

    fc->list[0].x = 0;

    fc->list[1].x = 1 << fc->rangebits;

    for (i = 2; i < fc->values; i++) {

        /*int a = i - 1;

        int g = ilog(a);

        assert(g <= fc->rangebits);

        a ^= 1 << (g-1);

        g = 1 << (fc->rangebits - g);

        fc->list[i].x = g + a*2*g;*/

        //int a[] = {14, 4, 58, 2, 8, 28, 90};

        int a[] = {93,23,372,6,46,186,750,14,33,65,130,260,556,3,10,18,28,39,55,79,111,158,220,312,464,650,850};

        fc->list[i].x = a[i - 2];

    }

    ready_floor(fc);



    venc->nresidues = 1;

    venc->residues = av_malloc(sizeof(residue_t) * venc->nresidues);



    // single residue

    rc = &venc->residues[0];

    rc->type = 2;

    rc->begin = 0;

    rc->end = 1600;

    rc->partition_size = 32;

    rc->classifications = 10;

    rc->classbook = 15;

    rc->books = av_malloc(sizeof(int[8]) * rc->classifications);

    for (i = 0; i < rc->classifications; i++) {

        int a[10][8] = {

            { -1, -1, -1, -1, -1, -1, -1, -1, },

            { -1, -1, 16, -1, -1, -1, -1, -1, },

            { -1, -1, 17, -1, -1, -1, -1, -1, },

            { -1, -1, 18, -1, -1, -1, -1, -1, },

            { -1, -1, 19, -1, -1, -1, -1, -1, },

            { -1, -1, 20, -1, -1, -1, -1, -1, },

            { -1, -1, 21, -1, -1, -1, -1, -1, },

            { 22, 23, -1, -1, -1, -1, -1, -1, },

            { 24, 25, -1, -1, -1, -1, -1, -1, },

            { 26, 27, 28, -1, -1, -1, -1, -1, },

        };

        int j;

        for (j = 0; j < 8; j++) rc->books[i][j] = a[i][j];

    }

    ready_residue(rc, venc);



    venc->nmappings = 1;

    venc->mappings = av_malloc(sizeof(mapping_t) * venc->nmappings);



    // single mapping

    mc = &venc->mappings[0];

    mc->submaps = 1;

    mc->mux = av_malloc(sizeof(int) * venc->channels);

    for (i = 0; i < venc->channels; i++) mc->mux[i] = 0;

    mc->floor = av_malloc(sizeof(int) * mc->submaps);

    mc->residue = av_malloc(sizeof(int) * mc->submaps);

    for (i = 0; i < mc->submaps; i++) {

        mc->floor[i] = 0;

        mc->residue[i] = 0;

    }

    mc->coupling_steps = venc->channels == 2 ? 1 : 0;

    mc->magnitude = av_malloc(sizeof(int) * mc->coupling_steps);

    mc->angle = av_malloc(sizeof(int) * mc->coupling_steps);

    if (mc->coupling_steps) {

        mc->magnitude[0] = 0;

        mc->angle[0] = 1;

    }



    venc->nmodes = 1;

    venc->modes = av_malloc(sizeof(vorbis_mode_t) * venc->nmodes);



    // single mode

    venc->modes[0].blockflag = 0;

    venc->modes[0].mapping = 0;



    venc->have_saved = 0;

    venc->saved = av_malloc(sizeof(float) * venc->channels * (1 << venc->blocksize[1]) / 2);

    venc->samples = av_malloc(sizeof(float) * venc->channels * (1 << venc->blocksize[1]));

    venc->floor = av_malloc(sizeof(float) * venc->channels * (1 << venc->blocksize[1]) / 2);

    venc->coeffs = av_malloc(sizeof(float) * venc->channels * (1 << venc->blocksize[1]) / 2);



    venc->win[0] = ff_vorbis_vwin[venc->blocksize[0] - 6];

    venc->win[1] = ff_vorbis_vwin[venc->blocksize[1] - 6];



    ff_mdct_init(&venc->mdct[0], venc->blocksize[0], 0);

    ff_mdct_init(&venc->mdct[1], venc->blocksize[1], 0);

}