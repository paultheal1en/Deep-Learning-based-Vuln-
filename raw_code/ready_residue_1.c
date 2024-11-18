static void ready_residue(vorbis_enc_residue *rc, vorbis_enc_context *venc)

{

    int i;

    assert(rc->type == 2);

    rc->maxes = av_mallocz(sizeof(float[2]) * rc->classifications);

    for (i = 0; i < rc->classifications; i++) {

        int j;

        vorbis_enc_codebook * cb;

        for (j = 0; j < 8; j++)

            if (rc->books[i][j] != -1)

                break;

        if (j == 8) // zero

            continue;

        cb = &venc->codebooks[rc->books[i][j]];

        assert(cb->ndimentions >= 2);

        assert(cb->lookup);



        for (j = 0; j < cb->nentries; j++) {

            float a;

            if (!cb->lens[j])

                continue;

            a = fabs(cb->dimentions[j * cb->ndimentions]);

            if (a > rc->maxes[i][0])

                rc->maxes[i][0] = a;

            a = fabs(cb->dimentions[j * cb->ndimentions + 1]);

            if (a > rc->maxes[i][1])

                rc->maxes[i][1] = a;

        }

    }

    // small bias

    for (i = 0; i < rc->classifications; i++) {

        rc->maxes[i][0] += 0.8;

        rc->maxes[i][1] += 0.8;

    }

}
