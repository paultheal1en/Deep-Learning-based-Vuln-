static av_always_inline int setup_classifs(vorbis_context *vc,

                                           vorbis_residue *vr,

                                           uint8_t *do_not_decode,

                                           unsigned ch_used,

                                           int partition_count)

{

    int p, j, i;

    unsigned c_p_c         = vc->codebooks[vr->classbook].dimensions;

    unsigned inverse_class = ff_inverse[vr->classifications];

    unsigned temp, temp2;

    for (p = 0, j = 0; j < ch_used; ++j) {

        if (!do_not_decode[j]) {

            temp = get_vlc2(&vc->gb, vc->codebooks[vr->classbook].vlc.table,

                                     vc->codebooks[vr->classbook].nb_bits, 3);



            av_dlog(NULL, "Classword: %u\n", temp);



            assert(vr->classifications > 1 && temp <= 65536); //needed for inverse[]



            for (i = 0; i < c_p_c; ++i) {

                temp2 = (((uint64_t)temp) * inverse_class) >> 32;

                if (partition_count + c_p_c - 1 - i < vr->ptns_to_read)

                    vr->classifs[p + partition_count + c_p_c - 1 - i] =

                        temp - temp2 * vr->classifications;

                temp = temp2;

            }

        }

        p += vr->ptns_to_read;

    }

    return 0;

}
