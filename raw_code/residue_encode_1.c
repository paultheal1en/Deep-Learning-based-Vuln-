static void residue_encode(vorbis_enc_context *venc, vorbis_enc_residue *rc,

                           PutBitContext *pb, float *coeffs, int samples,

                           int real_ch)

{

    int pass, i, j, p, k;

    int psize      = rc->partition_size;

    int partitions = (rc->end - rc->begin) / psize;

    int channels   = (rc->type == 2) ? 1 : real_ch;

    int classes[MAX_CHANNELS][NUM_RESIDUE_PARTITIONS];

    int classwords = venc->codebooks[rc->classbook].ndimentions;



    assert(rc->type == 2);

    assert(real_ch == 2);

    for (p = 0; p < partitions; p++) {

        float max1 = 0., max2 = 0.;

        int s = rc->begin + p * psize;

        for (k = s; k < s + psize; k += 2) {

            max1 = FFMAX(max1, fabs(coeffs[          k / real_ch]));

            max2 = FFMAX(max2, fabs(coeffs[samples + k / real_ch]));

        }



        for (i = 0; i < rc->classifications - 1; i++)

            if (max1 < rc->maxes[i][0] && max2 < rc->maxes[i][1])

                break;

        classes[0][p] = i;

    }



    for (pass = 0; pass < 8; pass++) {

        p = 0;

        while (p < partitions) {

            if (pass == 0)

                for (j = 0; j < channels; j++) {

                    vorbis_enc_codebook * book = &venc->codebooks[rc->classbook];

                    int entry = 0;

                    for (i = 0; i < classwords; i++) {

                        entry *= rc->classifications;

                        entry += classes[j][p + i];

                    }

                    put_codeword(pb, book, entry);

                }

            for (i = 0; i < classwords && p < partitions; i++, p++) {

                for (j = 0; j < channels; j++) {

                    int nbook = rc->books[classes[j][p]][pass];

                    vorbis_enc_codebook * book = &venc->codebooks[nbook];

                    float *buf = coeffs + samples*j + rc->begin + p*psize;

                    if (nbook == -1)

                        continue;



                    assert(rc->type == 0 || rc->type == 2);

                    assert(!(psize % book->ndimentions));



                    if (rc->type == 0) {

                        for (k = 0; k < psize; k += book->ndimentions) {

                            float *a = put_vector(book, pb, &buf[k]);

                            int l;

                            for (l = 0; l < book->ndimentions; l++)

                                buf[k + l] -= a[l];

                        }

                    } else {

                        int s = rc->begin + p * psize, a1, b1;

                        a1 = (s % real_ch) * samples;

                        b1 =  s / real_ch;

                        s  = real_ch * samples;

                        for (k = 0; k < psize; k += book->ndimentions) {

                            int dim, a2 = a1, b2 = b1;

                            float vec[MAX_CODEBOOK_DIM], *pv = vec;

                            for (dim = book->ndimentions; dim--; ) {

                                *pv++ = coeffs[a2 + b2];

                                if ((a2 += samples) == s) {

                                    a2 = 0;

                                    b2++;

                                }

                            }

                            pv = put_vector(book, pb, vec);

                            for (dim = book->ndimentions; dim--; ) {

                                coeffs[a1 + b1] -= *pv++;

                                if ((a1 += samples) == s) {

                                    a1 = 0;

                                    b1++;

                                }

                            }

                        }

                    }

                }

            }

        }

    }

}
