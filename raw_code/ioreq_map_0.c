static int ioreq_map(struct ioreq *ioreq)

{

    int gnt = ioreq->blkdev->xendev.gnttabdev;

    int i;



    if (ioreq->v.niov == 0) {

        return 0;

    }

    if (batch_maps) {

        ioreq->pages = xc_gnttab_map_grant_refs

            (gnt, ioreq->v.niov, ioreq->domids, ioreq->refs, ioreq->prot);

        if (ioreq->pages == NULL) {

            xen_be_printf(&ioreq->blkdev->xendev, 0,

                          "can't map %d grant refs (%s, %d maps)\n",

                          ioreq->v.niov, strerror(errno), ioreq->blkdev->cnt_map);

            return -1;

        }

        for (i = 0; i < ioreq->v.niov; i++) {

            ioreq->v.iov[i].iov_base = ioreq->pages + i * XC_PAGE_SIZE +

                (uintptr_t)ioreq->v.iov[i].iov_base;

        }

        ioreq->blkdev->cnt_map += ioreq->v.niov;

    } else  {

        for (i = 0; i < ioreq->v.niov; i++) {

            ioreq->page[i] = xc_gnttab_map_grant_ref

                (gnt, ioreq->domids[i], ioreq->refs[i], ioreq->prot);

            if (ioreq->page[i] == NULL) {

                xen_be_printf(&ioreq->blkdev->xendev, 0,

                              "can't map grant ref %d (%s, %d maps)\n",

                              ioreq->refs[i], strerror(errno), ioreq->blkdev->cnt_map);

                ioreq_unmap(ioreq);

                return -1;

            }

            ioreq->v.iov[i].iov_base = ioreq->page[i] + (uintptr_t)ioreq->v.iov[i].iov_base;

            ioreq->blkdev->cnt_map++;

        }

    }

    return 0;

}
