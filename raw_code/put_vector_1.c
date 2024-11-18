static float *put_vector(vorbis_enc_codebook *book, PutBitContext *pb,

                         float *num)

{

    int i, entry = -1;

    float distance = FLT_MAX;

    assert(book->dimentions);

    for (i = 0; i < book->nentries; i++) {

        float * vec = book->dimentions + i * book->ndimentions, d = book->pow2[i];

        int j;

        if (!book->lens[i])

            continue;

        for (j = 0; j < book->ndimentions; j++)

            d -= vec[j] * num[j];

        if (distance > d) {

            entry    = i;

            distance = d;

        }

    }

    put_codeword(pb, book, entry);

    return &book->dimentions[entry * book->ndimentions];

}
