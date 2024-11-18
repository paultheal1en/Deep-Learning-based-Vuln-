static void bwf_write_bext_chunk(AVFormatContext *s)

{

    AVDictionaryEntry *tmp_tag;

    uint64_t time_reference = 0;

    int64_t bext = ff_start_tag(s->pb, "bext");



    bwf_write_bext_string(s, "description", 256);

    bwf_write_bext_string(s, "originator", 32);

    bwf_write_bext_string(s, "originator_reference", 32);

    bwf_write_bext_string(s, "origination_date", 10);

    bwf_write_bext_string(s, "origination_time", 8);



    if (tmp_tag = av_dict_get(s->metadata, "time_reference", NULL, 0))

        time_reference = strtoll(tmp_tag->value, NULL, 10);

    avio_wl64(s->pb, time_reference);

    avio_wl16(s->pb, 1);  // set version to 1



    if (tmp_tag = av_dict_get(s->metadata, "umid", NULL, 0)) {

        unsigned char umidpart_str[17] = {0};

        int64_t i;

        uint64_t umidpart;

        size_t len = strlen(tmp_tag->value+2);



        for (i = 0; i < len/16; i++) {

            memcpy(umidpart_str, tmp_tag->value + 2 + (i*16), 16);

            umidpart = strtoll(umidpart_str, NULL, 16);

            avio_wb64(s->pb, umidpart);

        }

        ffio_fill(s->pb, 0, 64 - i*8);

    } else

        ffio_fill(s->pb, 0, 64); // zero UMID



    ffio_fill(s->pb, 0, 190); // Reserved



    if (tmp_tag = av_dict_get(s->metadata, "coding_history", NULL, 0))

        avio_put_str(s->pb, tmp_tag->value);



    ff_end_tag(s->pb, bext);

}
