static void ebml_free(EbmlSyntax *syntax, void *data)

{

    int i, j;

    for (i = 0; syntax[i].id; i++) {

        void *data_off = (char *) data + syntax[i].data_offset;

        switch (syntax[i].type) {

        case EBML_STR:

        case EBML_UTF8:

            av_freep(data_off);

            break;

        case EBML_BIN:

            av_freep(&((EbmlBin *) data_off)->data);

            break;

        case EBML_LEVEL1:

        case EBML_NEST:

            if (syntax[i].list_elem_size) {

                EbmlList *list = data_off;

                char *ptr = list->elem;

                for (j = 0; j < list->nb_elem;

                     j++, ptr += syntax[i].list_elem_size)

                    ebml_free(syntax[i].def.n, ptr);

                av_freep(&list->elem);


            } else

                ebml_free(syntax[i].def.n, data_off);

        default:

            break;

        }

    }

}