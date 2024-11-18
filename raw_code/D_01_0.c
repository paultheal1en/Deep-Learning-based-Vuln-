D(float, sse)

D(float, avx)

D(int16, mmx)

D(int16, sse2)



av_cold int swri_rematrix_init_x86(struct SwrContext *s){

#if HAVE_YASM

    int mm_flags = av_get_cpu_flags();

    int nb_in  = av_get_channel_layout_nb_channels(s->in_ch_layout);

    int nb_out = av_get_channel_layout_nb_channels(s->out_ch_layout);

    int num    = nb_in * nb_out;

    int i,j;



    s->mix_1_1_simd = NULL;

    s->mix_2_1_simd = NULL;



    if (s->midbuf.fmt == AV_SAMPLE_FMT_S16P){

        if(EXTERNAL_MMX(mm_flags)) {

            s->mix_1_1_simd = ff_mix_1_1_a_int16_mmx;

            s->mix_2_1_simd = ff_mix_2_1_a_int16_mmx;

        }

        if(EXTERNAL_SSE2(mm_flags)) {

            s->mix_1_1_simd = ff_mix_1_1_a_int16_sse2;

            s->mix_2_1_simd = ff_mix_2_1_a_int16_sse2;

        }

        s->native_simd_matrix = av_mallocz_array(num,  2 * sizeof(int16_t));

        s->native_simd_one    = av_mallocz(2 * sizeof(int16_t));

        if (!s->native_simd_matrix || !s->native_simd_one)

            return AVERROR(ENOMEM);



        for(i=0; i<nb_out; i++){

            int sh = 0;

            for(j=0; j<nb_in; j++)

                sh = FFMAX(sh, FFABS(((int*)s->native_matrix)[i * nb_in + j]));

            sh = FFMAX(av_log2(sh) - 14, 0);

            for(j=0; j<nb_in; j++) {

                ((int16_t*)s->native_simd_matrix)[2*(i * nb_in + j)+1] = 15 - sh;

                ((int16_t*)s->native_simd_matrix)[2*(i * nb_in + j)] =

                    ((((int*)s->native_matrix)[i * nb_in + j]) + (1<<sh>>1)) >> sh;

            }

        }

        ((int16_t*)s->native_simd_one)[1] = 14;

        ((int16_t*)s->native_simd_one)[0] = 16384;

    } else if(s->midbuf.fmt == AV_SAMPLE_FMT_FLTP){

        if(EXTERNAL_SSE(mm_flags)) {

            s->mix_1_1_simd = ff_mix_1_1_a_float_sse;

            s->mix_2_1_simd = ff_mix_2_1_a_float_sse;

        }

        if(EXTERNAL_AVX(mm_flags)) {

            s->mix_1_1_simd = ff_mix_1_1_a_float_avx;

            s->mix_2_1_simd = ff_mix_2_1_a_float_avx;

        }

        s->native_simd_matrix = av_mallocz_array(num, sizeof(float));

        s->native_simd_one = av_mallocz(sizeof(float));

        if (!s->native_simd_matrix || !s->native_simd_one)

            return AVERROR(ENOMEM);

        memcpy(s->native_simd_matrix, s->native_matrix, num * sizeof(float));

        memcpy(s->native_simd_one, s->native_one, sizeof(float));

    }

#endif



    return 0;

}
