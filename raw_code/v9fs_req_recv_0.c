static void v9fs_req_recv(P9Req *req, uint8_t id)

{

    QVirtIO9P *v9p = req->v9p;

    P9Hdr hdr;

    int i;



    for (i = 0; i < 10; i++) {

        qvirtio_wait_queue_isr(v9p->dev, v9p->vq, 1000 * 1000);



        v9fs_memread(req, &hdr, 7);

        le32_to_cpus(&hdr.size);

        le16_to_cpus(&hdr.tag);

        if (hdr.size >= 7) {

            break;

        }

        v9fs_memrewind(req, 7);

    }



    g_assert_cmpint(hdr.size, >=, 7);

    g_assert_cmpint(hdr.size, <=, P9_MAX_SIZE);

    g_assert_cmpint(hdr.tag, ==, req->tag);



    if (hdr.id != id) {

        g_printerr("Received response %d (%s) instead of %d (%s)\n",

                   hdr.id, rmessage_name(hdr.id), id, rmessage_name(id));



        if (hdr.id == P9_RLERROR) {

            uint32_t err;

            v9fs_uint32_read(req, &err);

            g_printerr("Rlerror has errno %d (%s)\n", err, strerror(err));

        }

    }

    g_assert_cmpint(hdr.id, ==, id);

}
