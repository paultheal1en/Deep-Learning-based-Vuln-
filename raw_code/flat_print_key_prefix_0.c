static void flat_print_key_prefix(WriterContext *wctx)

{

    FlatContext *flat = wctx->priv;

    const struct section *parent_section = wctx->section[wctx->level-1];



    printf("%s", flat->section_header[wctx->level].str);



    if (parent_section->flags & SECTION_FLAG_IS_ARRAY) {

        int n = parent_section->id == SECTION_ID_PACKETS_AND_FRAMES ?

            wctx->nb_section_packet_frame : wctx->nb_item[wctx->level-1];

        printf("%d%s", n, flat->sep_str);

    }

}
