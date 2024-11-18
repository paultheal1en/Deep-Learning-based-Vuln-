static int rd_strip(CinepakEncContext *s, int y, int h, int keyframe, AVPicture *last_pict, AVPicture *pict, AVPicture *scratch_pict, unsigned char *buf, int64_t *best_score)

{

    int64_t score = 0;

    int best_size = 0, v1_size, v4_size, v4, mb_count = s->w * h / MB_AREA;

    strip_info info;

    CinepakMode best_mode;

    int v4_codebooks[CODEBOOK_NUM][CODEBOOK_MAX*VECTOR_MAX];



    if(!keyframe)

        calculate_skip_errors(s, h, last_pict, pict, &info);



    //precompute V4 codebooks

    for(v4_size = 1, v4 = 0; v4_size <= 256; v4_size <<= 2, v4++) {

        info.v4_codebook = v4_codebooks[v4];

        quantize(s, h, pict, 0, v4_size, v4, &info);

    }



    //try all powers of 4 for the size of the codebooks

    //constraint the v4 codebook to be no bigger than the v1 codebook

    for(v1_size = 1; v1_size <= 256; v1_size <<= 2) {

        //compute V1 codebook

        quantize(s, h, pict, 1, v1_size, -1, &info);



        for(v4_size = 0, v4 = -1; v4_size <= v1_size; v4_size = v4_size ? v4_size << 2 : v1_size >= 4 ? v1_size >> 2 : 1, v4++) {

            //try all modes

            for(CinepakMode mode = 0; mode < MODE_COUNT; mode++) {

                //don't allow MODE_MC in inter frames

                if(keyframe && mode == MODE_MC)

                    continue;



                //only allow V1-only mode if v4 codebook is empty

                if(!v4_size && mode != MODE_V1_ONLY)

                    continue;



                info.v4_codebook = v4 >= 0 ? v4_codebooks[v4] : NULL;

                score = calculate_mode_score(s, mode, h, v1_size, v4_size, v4, &info);



                //av_log(s->avctx, AV_LOG_INFO, "%3i %3i score = %li\n", v1_size, v4_size, score);



                if(best_size == 0 || score < *best_score) {

                    *best_score = score;

                    best_size = encode_mode(s, mode, h, v1_size, v4_size, v4, scratch_pict, &info, s->strip_buf + STRIP_HEADER_SIZE);

                    best_mode = mode;



                    av_log(s->avctx, AV_LOG_INFO, "mode %i, %3i, %3i: %18li %i B\n", mode, v1_size, v4_size, score, best_size);



#ifdef CINEPAKENC_DEBUG

                    //save MB encoding choices

                    memcpy(s->best_mb, s->mb, mb_count*sizeof(mb_info));

#endif



                    //memcpy(strip_temp + STRIP_HEADER_SIZE, strip_temp, best_size);

                    write_strip_header(s, y, h, keyframe, s->strip_buf, best_size);

                }

            }

        }

    }



#ifdef CINEPAKENC_DEBUG

    //gather stats. this will only work properly of MAX_STRIPS == 1

    if(best_mode == MODE_V1_ONLY) {

        s->num_v1_mode++;

        s->num_v1_encs += s->w*h/MB_AREA;

    } else {

        if(best_mode == MODE_V1_V4)

            s->num_v4_mode++;

        else

            s->num_mc_mode++;



        int x;

        for(x = 0; x < s->w*h/MB_AREA; x++)

            if(s->best_mb[x].best_encoding == ENC_V1)

                s->num_v1_encs++;

            else if(s->best_mb[x].best_encoding == ENC_V4)

                s->num_v4_encs++;

            else

                s->num_skips++;

    }

#endif



    best_size += STRIP_HEADER_SIZE;

    memcpy(buf, s->strip_buf, best_size);





    return best_size;

}
