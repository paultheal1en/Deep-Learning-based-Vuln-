static int vorbis_parse_setup_hdr_mappings(vorbis_context *vc) {

    GetBitContext *gb=&vc->gb;

    uint_fast8_t i, j;



    vc->mapping_count=get_bits(gb, 6)+1;

    vc->mappings=av_mallocz(vc->mapping_count * sizeof(vorbis_mapping));



    AV_DEBUG(" There are %d mappings. \n", vc->mapping_count);



    for(i=0;i<vc->mapping_count;++i) {

        vorbis_mapping *mapping_setup=&vc->mappings[i];



        if (get_bits(gb, 16)) {

            av_log(vc->avccontext, AV_LOG_ERROR, "Other mappings than type 0 are not compliant with the Vorbis I specification. \n");

            return 1;

        }

        if (get_bits1(gb)) {

            mapping_setup->submaps=get_bits(gb, 4)+1;

        } else {

            mapping_setup->submaps=1;

        }



        if (get_bits1(gb)) {

            mapping_setup->coupling_steps=get_bits(gb, 8)+1;

            mapping_setup->magnitude=av_mallocz(mapping_setup->coupling_steps * sizeof(uint_fast8_t));

            mapping_setup->angle    =av_mallocz(mapping_setup->coupling_steps * sizeof(uint_fast8_t));

            for(j=0;j<mapping_setup->coupling_steps;++j) {

                mapping_setup->magnitude[j]=get_bits(gb, ilog(vc->audio_channels-1));

                mapping_setup->angle[j]=get_bits(gb, ilog(vc->audio_channels-1));

                // FIXME: sanity checks

            }

        } else {

            mapping_setup->coupling_steps=0;

        }



        AV_DEBUG("   %d mapping coupling steps: %d \n", i, mapping_setup->coupling_steps);



        if(get_bits(gb, 2)) {

            av_log(vc->avccontext, AV_LOG_ERROR, "%d. mapping setup data invalid. \n", i);

            return 1; // following spec.

        }



        if (mapping_setup->submaps>1) {

            mapping_setup->mux=av_mallocz(vc->audio_channels * sizeof(uint_fast8_t));

            for(j=0;j<vc->audio_channels;++j) {

                mapping_setup->mux[j]=get_bits(gb, 4);

            }

        }



        for(j=0;j<mapping_setup->submaps;++j) {

            skip_bits(gb, 8); // FIXME check?

            mapping_setup->submap_floor[j]=get_bits(gb, 8);

            mapping_setup->submap_residue[j]=get_bits(gb, 8);



            AV_DEBUG("   %d mapping %d submap : floor %d, residue %d \n", i, j, mapping_setup->submap_floor[j], mapping_setup->submap_residue[j]);

        }

    }

    return 0;

}
