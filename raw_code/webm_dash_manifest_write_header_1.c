static int webm_dash_manifest_write_header(AVFormatContext *s)

{

    int i;

    double start = 0.0;

    WebMDashMuxContext *w = s->priv_data;

    parse_adaptation_sets(s);

    write_header(s);

    avio_printf(s->pb, "<Period id=\"0\"");

    avio_printf(s->pb, " start=\"PT%gS\"", start);

    if (!w->is_live) {

        avio_printf(s->pb, " duration=\"PT%gS\"", get_duration(s));

    }

    avio_printf(s->pb, " >\n");



    for (i = 0; i < w->nb_as; i++) {

        if (write_adaptation_set(s, i) < 0) return -1;

    }



    avio_printf(s->pb, "</Period>\n");

    write_footer(s);

    return 0;

}
