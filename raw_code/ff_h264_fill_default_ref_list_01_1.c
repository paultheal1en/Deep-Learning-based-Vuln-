int ff_h264_fill_default_ref_list(H264Context *h)

{

    int i, len;



    if (h->slice_type_nos == AV_PICTURE_TYPE_B) {

        Picture *sorted[32];

        int cur_poc, list;

        int lens[2];



        if (FIELD_PICTURE(h))

            cur_poc = h->cur_pic_ptr->field_poc[h->picture_structure == PICT_BOTTOM_FIELD];

        else

            cur_poc = h->cur_pic_ptr->poc;



        for (list = 0; list < 2; list++) {

            len  = add_sorted(sorted,       h->short_ref, h->short_ref_count, cur_poc, 1 ^ list);

            len += add_sorted(sorted + len, h->short_ref, h->short_ref_count, cur_poc, 0 ^ list);

            assert(len <= 32);

            len  = build_def_list(h->default_ref_list[list],       sorted,      len, 0, h->picture_structure);

            len += build_def_list(h->default_ref_list[list] + len, h->long_ref, 16,  1, h->picture_structure);

            assert(len <= 32);



            if (len < h->ref_count[list])

                memset(&h->default_ref_list[list][len], 0, sizeof(Picture) * (h->ref_count[list] - len));

            lens[list] = len;

        }



        if (lens[0] == lens[1] && lens[1] > 1) {

            for (i = 0; i < lens[0] &&

                        h->default_ref_list[0][i].f.buf[0]->buffer ==

                        h->default_ref_list[1][i].f.buf[0]->buffer; i++);

            if (i == lens[0]) {

                Picture tmp;

                COPY_PICTURE(&tmp, &h->default_ref_list[1][0]);

                COPY_PICTURE(&h->default_ref_list[1][0], &h->default_ref_list[1][1]);

                COPY_PICTURE(&h->default_ref_list[1][1], &tmp);

            }

        }

    } else {

        len  = build_def_list(h->default_ref_list[0],       h->short_ref, h->short_ref_count, 0, h->picture_structure);

        len += build_def_list(h->default_ref_list[0] + len, h-> long_ref, 16,                 1, h->picture_structure);

        assert(len <= 32);

        if (len < h->ref_count[0])

            memset(&h->default_ref_list[0][len], 0, sizeof(Picture) * (h->ref_count[0] - len));

    }

#ifdef TRACE

    for (i = 0; i < h->ref_count[0]; i++) {

        tprintf(h->avctx, "List0: %s fn:%d 0x%p\n",

                (h->default_ref_list[0][i].long_ref ? "LT" : "ST"),

                h->default_ref_list[0][i].pic_id,

                h->default_ref_list[0][i].f.data[0]);

    }

    if (h->slice_type_nos == AV_PICTURE_TYPE_B) {

        for (i = 0; i < h->ref_count[1]; i++) {

            tprintf(h->avctx, "List1: %s fn:%d 0x%p\n",

                    (h->default_ref_list[1][i].long_ref ? "LT" : "ST"),

                    h->default_ref_list[1][i].pic_id,

                    h->default_ref_list[1][i].f.data[0]);

        }

    }

#endif

    return 0;

}
