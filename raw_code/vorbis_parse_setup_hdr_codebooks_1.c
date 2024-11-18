static int vorbis_parse_setup_hdr_codebooks(vorbis_context *vc) {

    uint_fast16_t cb;

    uint8_t *tmp_vlc_bits;

    uint32_t *tmp_vlc_codes;

    GetBitContext *gb=&vc->gb;



    vc->codebook_count=get_bits(gb,8)+1;



    AV_DEBUG(" Codebooks: %d \n", vc->codebook_count);



    vc->codebooks=(vorbis_codebook *)av_mallocz(vc->codebook_count * sizeof(vorbis_codebook));

    tmp_vlc_bits=(uint8_t *)av_mallocz(V_MAX_VLCS * sizeof(uint8_t));

    tmp_vlc_codes=(uint32_t *)av_mallocz(V_MAX_VLCS * sizeof(uint32_t));



    for(cb=0;cb<vc->codebook_count;++cb) {

        vorbis_codebook *codebook_setup=&vc->codebooks[cb];

        uint_fast8_t ordered;

        uint_fast32_t t, used_entries=0;

        uint_fast32_t entries;



        AV_DEBUG(" %d. Codebook \n", cb);



        if (get_bits(gb, 24)!=0x564342) {

            av_log(vc->avccontext, AV_LOG_ERROR, " %"PRIdFAST16". Codebook setup data corrupt. \n", cb);

            goto error;

        }



        codebook_setup->dimensions=get_bits(gb, 16);

        if (codebook_setup->dimensions>16) {

            av_log(vc->avccontext, AV_LOG_ERROR, " %"PRIdFAST16". Codebook's dimension is too large (%d). \n", cb, codebook_setup->dimensions);

            goto error;

        }

        entries=get_bits(gb, 24);

        if (entries>V_MAX_VLCS) {

            av_log(vc->avccontext, AV_LOG_ERROR, " %"PRIdFAST16". Codebook has too many entries (%"PRIdFAST32"). \n", cb, entries);

            goto error;

        }



        ordered=get_bits1(gb);



        AV_DEBUG(" codebook_dimensions %d, codebook_entries %d \n", codebook_setup->dimensions, entries);



        if (!ordered) {

            uint_fast16_t ce;

            uint_fast8_t flag;

            uint_fast8_t sparse=get_bits1(gb);



            AV_DEBUG(" not ordered \n");



            if (sparse) {

                AV_DEBUG(" sparse \n");



                used_entries=0;

                for(ce=0;ce<entries;++ce) {

                    flag=get_bits1(gb);

                    if (flag) {

                        tmp_vlc_bits[ce]=get_bits(gb, 5)+1;

                        ++used_entries;

                    }

                    else tmp_vlc_bits[ce]=0;

                }

            } else {

                AV_DEBUG(" not sparse \n");



                used_entries=entries;

                for(ce=0;ce<entries;++ce) {

                    tmp_vlc_bits[ce]=get_bits(gb, 5)+1;

                }

            }

        } else {

            uint_fast16_t current_entry=0;

            uint_fast8_t current_length=get_bits(gb, 5)+1;



            AV_DEBUG(" ordered, current length: %d \n", current_length);  //FIXME



            used_entries=entries;

            for(;current_entry<used_entries;++current_length) {

                uint_fast16_t i, number;



                AV_DEBUG(" number bits: %d ", ilog(entries - current_entry));



                number=get_bits(gb, ilog(entries - current_entry));



                AV_DEBUG(" number: %d \n", number);



                for(i=current_entry;i<number+current_entry;++i) {

                    if (i<used_entries) tmp_vlc_bits[i]=current_length;

                }



                current_entry+=number;

            }

            if (current_entry>used_entries) {

                av_log(vc->avccontext, AV_LOG_ERROR, " More codelengths than codes in codebook. \n");

                goto error;

            }

        }



        codebook_setup->lookup_type=get_bits(gb, 4);



        AV_DEBUG(" lookup type: %d : %s \n", codebook_setup->lookup_type, codebook_setup->lookup_type ? "vq" : "no lookup" );



// If the codebook is used for (inverse) VQ, calculate codevectors.



        if (codebook_setup->lookup_type==1) {

            uint_fast16_t i, j, k;

            uint_fast16_t codebook_lookup_values=ff_vorbis_nth_root(entries, codebook_setup->dimensions);

            uint_fast16_t codebook_multiplicands[codebook_lookup_values];



            float codebook_minimum_value=vorbisfloat2float(get_bits_long(gb, 32));

            float codebook_delta_value=vorbisfloat2float(get_bits_long(gb, 32));

            uint_fast8_t codebook_value_bits=get_bits(gb, 4)+1;

            uint_fast8_t codebook_sequence_p=get_bits1(gb);



            AV_DEBUG(" We expect %d numbers for building the codevectors. \n", codebook_lookup_values);

            AV_DEBUG("  delta %f minmum %f \n", codebook_delta_value, codebook_minimum_value);



            for(i=0;i<codebook_lookup_values;++i) {

                codebook_multiplicands[i]=get_bits(gb, codebook_value_bits);



                AV_DEBUG(" multiplicands*delta+minmum : %e \n", (float)codebook_multiplicands[i]*codebook_delta_value+codebook_minimum_value);

                AV_DEBUG(" multiplicand %d \n", codebook_multiplicands[i]);

            }



// Weed out unused vlcs and build codevector vector

            codebook_setup->codevectors=(float *)av_mallocz(used_entries*codebook_setup->dimensions * sizeof(float));

            for(j=0, i=0;i<entries;++i) {

                uint_fast8_t dim=codebook_setup->dimensions;



                if (tmp_vlc_bits[i]) {

                    float last=0.0;

                    uint_fast32_t lookup_offset=i;



#ifdef V_DEBUG

                    av_log(vc->avccontext, AV_LOG_INFO, "Lookup offset %d ,", i);

#endif



                    for(k=0;k<dim;++k) {

                        uint_fast32_t multiplicand_offset = lookup_offset % codebook_lookup_values;

                        codebook_setup->codevectors[j*dim+k]=codebook_multiplicands[multiplicand_offset]*codebook_delta_value+codebook_minimum_value+last;

                        if (codebook_sequence_p) {

                            last=codebook_setup->codevectors[j*dim+k];

                        }

                        lookup_offset/=codebook_lookup_values;

                    }

                    tmp_vlc_bits[j]=tmp_vlc_bits[i];



#ifdef V_DEBUG

                    av_log(vc->avccontext, AV_LOG_INFO, "real lookup offset %d, vector: ", j);

                    for(k=0;k<dim;++k) {

                        av_log(vc->avccontext, AV_LOG_INFO, " %f ", codebook_setup->codevectors[j*dim+k]);

                    }

                    av_log(vc->avccontext, AV_LOG_INFO, "\n");

#endif



                    ++j;

                }

            }

            if (j!=used_entries) {

                av_log(vc->avccontext, AV_LOG_ERROR, "Bug in codevector vector building code. \n");

                goto error;

            }

            entries=used_entries;

        }

        else if (codebook_setup->lookup_type>=2) {

            av_log(vc->avccontext, AV_LOG_ERROR, "Codebook lookup type not supported. \n");

            goto error;

        }



// Initialize VLC table

        if (ff_vorbis_len2vlc(tmp_vlc_bits, tmp_vlc_codes, entries)) {

            av_log(vc->avccontext, AV_LOG_ERROR, " Invalid code lengths while generating vlcs. \n");

            goto error;

        }

        codebook_setup->maxdepth=0;

        for(t=0;t<entries;++t)

            if (tmp_vlc_bits[t]>=codebook_setup->maxdepth) codebook_setup->maxdepth=tmp_vlc_bits[t];



        if(codebook_setup->maxdepth > 3*V_NB_BITS) codebook_setup->nb_bits=V_NB_BITS2;

        else                                       codebook_setup->nb_bits=V_NB_BITS;



        codebook_setup->maxdepth=(codebook_setup->maxdepth+codebook_setup->nb_bits-1)/codebook_setup->nb_bits;



        if (init_vlc(&codebook_setup->vlc, codebook_setup->nb_bits, entries, tmp_vlc_bits, sizeof(*tmp_vlc_bits), sizeof(*tmp_vlc_bits), tmp_vlc_codes, sizeof(*tmp_vlc_codes), sizeof(*tmp_vlc_codes), INIT_VLC_LE)) {

            av_log(vc->avccontext, AV_LOG_ERROR, " Error generating vlc tables. \n");

            goto error;

        }

    }



    av_free(tmp_vlc_bits);

    av_free(tmp_vlc_codes);

    return 0;



// Error:

error:

    av_free(tmp_vlc_bits);

    av_free(tmp_vlc_codes);

    return 1;

}
