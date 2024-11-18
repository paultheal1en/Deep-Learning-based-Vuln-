void ff_h264_remove_all_refs(H264Context *h)

{

    int i;



    for (i = 0; i < 16; i++) {

        remove_long(h, i, 0);

    }

    assert(h->long_ref_count == 0);



    if (h->short_ref_count && !h->last_pic_for_ec.f->data[0]) {

        ff_h264_unref_picture(h, &h->last_pic_for_ec);

        if (h->short_ref[0]->f->buf[0])

            ff_h264_ref_picture(h, &h->last_pic_for_ec, h->short_ref[0]);

    }



    for (i = 0; i < h->short_ref_count; i++) {

        unreference_pic(h, h->short_ref[i], 0);

        h->short_ref[i] = NULL;

    }

    h->short_ref_count = 0;



    memset(h->default_ref, 0, sizeof(h->default_ref));

}
