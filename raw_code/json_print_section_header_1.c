static void json_print_section_header(WriterContext *wctx)

{

    JSONContext *json = wctx->priv;

    AVBPrint buf;

    const struct section *section = wctx->section[wctx->level];

    const struct section *parent_section = wctx->level ?

        wctx->section[wctx->level-1] : NULL;



    if (wctx->level && wctx->nb_item[wctx->level-1])

        printf(",\n");



    if (section->flags & SECTION_FLAG_IS_WRAPPER) {

        printf("{\n");

        json->indent_level++;

    } else {

        av_bprint_init(&buf, 1, AV_BPRINT_SIZE_UNLIMITED);

        json_escape_str(&buf, section->name, wctx);

        JSON_INDENT();



        json->indent_level++;

        if (section->flags & SECTION_FLAG_IS_ARRAY) {

            printf("\"%s\": [\n", buf.str);

        } else if (!(parent_section->flags & SECTION_FLAG_IS_ARRAY)) {

            printf("\"%s\": {%s", buf.str, json->item_start_end);

        } else {

            printf("{%s", json->item_start_end);



            /* this is required so the parser can distinguish between packets and frames */

            if (parent_section->id == SECTION_ID_PACKETS_AND_FRAMES) {

                if (!json->compact)

                    JSON_INDENT();

                printf("\"type\": \"%s\"%s", section->name, json->item_sep);

            }

        }

        av_bprint_finalize(&buf, NULL);

    }

}
