static void ready_codebook(vorbis_enc_codebook *cb)

{

    int i;



    ff_vorbis_len2vlc(cb->lens, cb->codewords, cb->nentries);



    if (!cb->lookup) {

        cb->pow2 = cb->dimentions = NULL;

    } else {

        int vals = cb_lookup_vals(cb->lookup, cb->ndimentions, cb->nentries);

        cb->dimentions = av_malloc(sizeof(float) * cb->nentries * cb->ndimentions);

        cb->pow2 = av_mallocz(sizeof(float) * cb->nentries);

        for (i = 0; i < cb->nentries; i++) {

            float last = 0;

            int j;

            int div = 1;

            for (j = 0; j < cb->ndimentions; j++) {

                int off;

                if (cb->lookup == 1)

                    off = (i / div) % vals; // lookup type 1

                else

                    off = i * cb->ndimentions + j; // lookup type 2



                cb->dimentions[i * cb->ndimentions + j] = last + cb->min + cb->quantlist[off] * cb->delta;

                if (cb->seq_p)

                    last = cb->dimentions[i * cb->ndimentions + j];

                cb->pow2[i] += cb->dimentions[i * cb->ndimentions + j] * cb->dimentions[i * cb->ndimentions + j];

                div *= vals;

            }

            cb->pow2[i] /= 2.;

        }

    }

}
